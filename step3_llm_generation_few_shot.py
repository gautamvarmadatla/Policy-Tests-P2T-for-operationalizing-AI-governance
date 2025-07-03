# =========================
# Structured Extraction (Step 3) — End-to-End Notebook Cell
# =========================
# What it does:
# - Reads candidate clause JSONL (from Step 2 mining)
# - Calls OpenAI GPT (reasoning/4o family preferred) with a strict system prompt + few-shots
# - Enforces array-of-rules JSON schema, normalizes, and validates
# - Writes:
#     1) policy_schema.yaml   (list of rules; falls back to .json if PyYAML not available)
#     2) <basename>.extracted.jsonl  (per-candidate results: rules or error with provenance)
#
# Usage (example at bottom of cell):
#   - Ensure OPENAI_API_KEY env var is set
#   - Set paths to your candidates JSONL
#   - Run run_hardened(...)

from __future__ import annotations
import os, sys, json, re, time, pathlib, argparse
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path  # <-- added so _fallback_doc_and_cit can use Path
import hashlib
import logging
from urllib.parse import urlparse

_level_env = (os.getenv("PIPELINE_LOG_LEVEL", "INFO") or "INFO").strip().upper()
_level = logging._nameToLevel.get(_level_env, logging.INFO)  # fallback to INFO if unknown

logger = logging.getLogger("policy_pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


try:
    import tldextract
    _HAVE_TLDX = True
except Exception:
    _HAVE_TLDX = False

try:
    import z3 as _z3  # presence probe only
    HAVE_Z3 = True
except Exception:
    HAVE_Z3 = False

# ---- Optional deps handling (PyYAML, jsonschema, openai) ----
import importlib, subprocess
def _ensure_pkg(pkg_spec: str):
    name = pkg_spec.split("==")[0].split(">=")[0]
    try:
        importlib.import_module(name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_spec])

for _p in ["openai>=1.0.0", "jsonschema>=4.21.0", "pyyaml>=6.0.0", "z3-solver>=4.12.2"]:
    try:
        importlib.import_module(_p.split(">=")[0])
    except Exception:
        try:
            _ensure_pkg(_p)
        except Exception as _e:
            print(f"Warning: could not install {_p}: {_e}")

# Now import
try:
    from jsonschema import Draft202012Validator, ValidationError
except Exception as e:
    raise RuntimeError("jsonschema is required. Try: pip install jsonschema>=4.21.0") from e

try:
    import yaml
    HAVE_YAML = True
except Exception:
    yaml = None
    HAVE_YAML = False


# Optional: z3 for global consistency checks (used if present)
_PRED_ENUM = None  # (PredState, Unknown, Require, Forbid)

def _get_pred_enum():
    global _PRED_ENUM
    if _PRED_ENUM is None:
        from z3 import EnumSort
        _PRED_ENUM = EnumSort("PredState", ["Unknown", "Require", "Forbid"])
    return _PRED_ENUM

def _host_etld1(url: str) -> Optional[str]:
    """
    Return a normalized host string for allow-list checks.
    If tldextract is available, prefer eTLD+1 (e.g., sub.a.b.nih.gov -> nih.gov).
    Otherwise, return the hostname (e.g., sub.a.b.nih.gov).
    """
    try:
        host = urlparse(url).hostname or ""
        if not host:
            return None
        host = host.lower().strip(".")
        if _HAVE_TLDX:
            ex = tldextract.extract(host)
            # ex.domain == "nih", ex.suffix == "gov" -> "nih.gov"
            if ex.domain and ex.suffix:
                return f"{ex.domain}.{ex.suffix}".lower()
        return host
    except Exception:
        return None

def _stable_id_prefix(doc: str, span_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (doc or "doc").lower()).strip("-")[:40]
    leaf = (span_id.split("/")[-1] or "span")
    return f"{slug}/{leaf}"

def _rule_hash(r: dict) -> str:
    # Include fields that uniquely characterize the rule’s meaning and provenance
    payload = {
        "req": r.get("requirements", []),
        "haz": r.get("hazard", ""),
        "scope": r.get("scope", {}),
        "src_doc": (r.get("source", {}) or {}).get("doc", ""),
        "src_cit": (r.get("source", {}) or {}).get("citation", ""),
        "src_span": (r.get("source", {}) or {}).get("span_id", ""),
        "exceptions": r.get("exceptions", []),
        "conditions": r.get("conditions", []),
        "severity": r.get("severity", ""),
    }
    j = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(j.encode("utf-8")).hexdigest()[:8]


def _fallback_doc_and_cit(cand: dict, *, candidates_jsonl_path: str) -> tuple[str, str]:
    """
    Derive a document title and a best-effort citation when Step 2 didn’t populate them.
    - doc: prefer cand['doc'] else PDF stem else first folder of span_id
    - citation: prefer cand['citation'] else "" (or a section-like hint from span_id)
    """
    doc = (cand.get("doc") or "").strip()
    cit = (cand.get("citation") or "").strip()

    if not doc:
        # 1) Try PDF name from the candidates file location
        try:
            # e.g., "<repo>/out/NIST.AI.100-1/spans.jsonl" ⇒ "NIST.AI.100-1"
            doc = Path(candidates_jsonl_path).parent.parent.name
            if not doc or doc.lower() in {"out", "tmp", "artifacts"}:
                # Fallback to the .jsonl file’s stem
                doc = Path(candidates_jsonl_path).stem.replace(".candidates", "")
        except Exception:
            doc = ""

    # 2) If still empty, try the span_id’s first segment
    span_id = (cand.get("span_id") or "").strip()
    if not doc and span_id:
        parts = span_id.split("/", 1)
        if parts and parts[0]:
            doc = parts[0]

    # 3) Optional: derive a tiny section-ish hint for citation from span path
    if not cit and span_id:
        # Pull last 1–2 path bits to look like a “section”
        bits = span_id.split("/")
        if len(bits) >= 2:
            cit = "/".join(bits[1:3])

    return (doc or ""), (cit or "")



# ---- Canonical vocab & synonym maps ----
# ---- Canonical vocab & synonym maps (90% coverage target) ----
# ===== Canonical enums (v2) =====
# Keep existing tokens; add high-signal, broadly useful ones.
ACTOR_ENUM = {
    # Original 10
    "user","model","agent","org","provider","deployer","developer",
    "evaluator","admin","auditor",
    # Added (governance/ops/ownership/roles)
    "security","legal","privacy","risk","data_owner","data_steward",
    "maintainer","analyst","pm","support", "entity" , "business associate", 
    "operator" , "practitioner", "supplier" , "vendor"
}

DOMAIN_ENUM = {
    # Original
    "PHI","PII","health","code","text","image","audio","biometric","general",
    # Added (regulatory + operationally testable)
    "financial",      # bank acct, payments, transactions, payroll
    "credentials",    # passwords, tokens, API keys (non-human secrets go to secrets)
    "secrets",        # private keys, signing keys, KMS materials
    "geolocation",    # GPS, cell/wifi location, precise loc
    "telemetry",      # logs/metrics/traces incl. analytics events
    "network",        # IPs, headers, user agents, device IDs (if not PII-labeled)
    "video",          # distinct from image
    "document",       # office/PDF/contract-like artifacts
    "education",      # student education records (FERPA-style)
    "children",       # data about children/minors (COPPA-like)
    "trade_secret"    # confidential biz data (non-code, non-secrets)
}

CONTEXT_ENUM = {
    # Original
    "prod","staging","eval","tenant","repo","high-risk","public","internal","research","training","finetune",
    # Added (envs, deploy surfaces, runtime modes)
    "dev","ci","qa",          # full SDLC coverage
    "inference",              # distinct from training/finetune/eval
    "onprem","cloud","edge",  # deployment surfaces
    "batch","stream",         # data/runtime modes
    "regulated"               # broad regulatory designation (separate from high-risk)
}


CANON = {
  "actor": {
    # org-level
    "organization":"org","organisation":"org","company":"org","enterprise":"org","employer":"org",
    "controller":"org","data controller":"org","owner":"org","product owner":"org",
    # service/ops
    "processor":"provider","data processor":"provider","operator":"provider","service provider":"provider",
    "vendor":"provider","third party":"provider","third-party":"provider","subprocessor":"provider",
    "platform":"provider","host":"provider","ops":"provider","operations":"provider","site reliability":"provider","sre":"provider",
    # deploy/run
    "devops":"deployer","site reliability engineer":"deployer","release engineer":"deployer",
    "deployment":"deployer","deployment engineer":"deployer","infra":"deployer","infrastructure":"deployer",
    # build/dev
    "developer":"developer","engineer":"developer","ml engineer":"developer","ml-engineer":"developer",
    "data scientist":"developer","research engineer":"developer","software engineer":"developer","programmer":"developer",
    # model/agents
    "system":"model","ai system":"model","llm":"model","model":"model","assistant":"agent",
    "chatbot":"agent","bot":"agent","tool agent":"agent","autonomous agent":"agent","agentic system":"agent",
    # evaluation/review
    "evaluator":"evaluator","annotator":"evaluator","rater":"evaluator","human reviewer":"evaluator",
    "content reviewer":"evaluator","qa":"evaluator","quality assurance":"evaluator","tester":"evaluator",
    # admin/governance
    "admin":"admin","administrator":"admin","policy admin":"admin","moderator":"admin",
    "security admin":"admin","compliance admin":"admin",
    # audit/compliance
    "auditor":"auditor","compliance officer":"auditor","privacy officer":"auditor",
    "dpo":"auditor","data protection officer":"auditor","risk officer":"auditor",
    # users
    "end user":"user","end-user":"user","customer":"user","client":"user","consumer":"user","patient":"user","subject":"user"
  },

  "data_domain": {
    # PHI / health
    "phi":"PHI","protected health information":"PHI","health information":"PHI","medical record":"PHI",
    # PII / identifiers
    "pii":"PII","personally identifiable information":"PII","personal data":"PII","personal information":"PII",
    "customer data":"PII","user data":"PII","contact info":"PII","contact information":"PII",
    "government id":"PII","government-id":"PII","gov id":"PII","ssn":"PII","social security number":"PII",
    "passport":"PII","driver license":"PII","driver’s license":"PII","drivers license":"PII",
    "email":"PII","phone":"PII","phone number":"PII","address":"PII","ip address":"PII","device id":"PII",
    "payment card":"PII","credit card":"PII","cardholder data":"PII","pci":"PII",
    # health (non-PHI context / general medical)
    "health":"health","clinical data":"health","medical data":"health","diagnosis data":"health","genomic data":"health",
    # code/text/media
    "source code":"code","software code":"code","repository code":"code","binary code":"code","snippet":"code","patch":"code",
    "text":"text","plaintext":"text","document text":"text",
    "image":"image","photo":"image","screenshot":"image","diagram":"image","figure":"image",
    "audio":"audio","voice":"audio","recording":"audio",
    "biometric":"biometric","biometric identifiers":"biometric","faceprint":"biometric","fingerprint":"biometric","iris":"biometric",
    # catch-all
    "data":"general","metadata":"general","logs":"general","telemetry":"general","analytics":"general"
  },

  "context": {
    # prod / staging / eval
    "production":"prod","prod":"prod","live":"prod","runtime":"prod","customer-facing":"prod","internet-facing":"prod",
    "preprod":"staging","pre-production":"staging","preproduction":"staging","stage":"staging","staging":"staging","canary":"staging",
    "uat":"staging","user acceptance":"staging",
    "test":"eval","testing":"eval","evaluation":"eval","offline eval":"eval","ab test":"eval","a/b test":"eval","experimentation":"eval",
    # tenancy / boundary
    "multitenant":"tenant","multi-tenant":"tenant","tenant boundary":"tenant","tenant":"tenant","workspace boundary":"tenant",
    "cross-tenant":"tenant","cross tenant":"tenant","org boundary":"tenant","customer boundary":"tenant",
    # repos / scm
    "repository":"repo","repo":"repo","git":"repo","github":"repo","gitlab":"repo","bitbucket":"repo","pull request":"repo","merge request":"repo",
    # risk level
    "high risk":"high-risk","high-risk":"high-risk","safety-critical":"high-risk","regulated":"high-risk","high impact":"high-risk","sensitive":"high-risk",
    # exposure
    "public web":"public","public":"public","external":"public","externally accessible":"public",
    "internal":"internal","private":"internal","intranet":"internal","employee-only":"internal","authenticated-only":"internal",
    # r&d / training / finetune
    "research":"research","r&d":"research","lab":"research",
    "training":"training","pretraining":"training","pre-training":"training",
    "fine tune":"finetune","fine-tune":"finetune","fine tuning":"finetune","finetuning":"finetune","finetune":"finetune"
  }
}

# ===== Additions to CANON["actor"] =====
CANON["actor"].update({
    # governance/ops roles
    "secops":"security","security":"security","security team":"security","appsec":"security",
    "ciso":"security","security engineer":"security","security analyst":"security",
    "privacy":"privacy","privacy team":"privacy","dpo":"privacy","data protection officer":"privacy",
    "compliance":"auditor","compliance officer":"auditor","risk":"risk","risk officer":"risk",
    "legal":"legal","counsel":"legal","legal team":"legal",
    "data owner":"data_owner","owner of data":"data_owner","business owner":"data_owner",
    "data steward":"data_steward","steward":"data_steward",
    "maintainer":"maintainer","site reliability engineer":"maintainer","sre":"maintainer",
    "analyst":"analyst","data analyst":"analyst","security analyst":"security",  # prefer security
    "product manager":"pm","pm":"pm",
    "support":"support","customer support":"support","helpdesk":"support"
})

# ===== Additions to CANON["data_domain"] =====
CANON["data_domain"].update({
    # financial
    "financial":"financial","finance":"financial","payment":"financial","payments":"financial",
    "bank":"financial","banking":"financial","iban":"financial","swift":"financial","ach":"financial",
    "payroll":"financial","transaction":"financial","txn":"financial",
    # credentials / secrets
    "credential":"credentials","credentials":"credentials","password":"credentials",
    "api key":"credentials","api keys":"credentials","token":"credentials","session token":"credentials",
    "secret":"secrets","private key":"secrets","signing key":"secrets","kms key":"secrets",
    # geolocation
    "location":"geolocation","geo":"geolocation","gps":"geolocation","latitude":"geolocation","longitude":"geolocation",
    # telemetry / logs / network
    "telemetry":"telemetry","metric":"telemetry","metrics":"telemetry","trace":"telemetry","traces":"telemetry",
    "log":"telemetry","logs":"telemetry","analytics":"telemetry","event data":"telemetry",
    "network":"network","network data":"network","network log":"network","user agent":"network",
    # media / docs
    "video":"video","videos":"video","document":"document","docs":"document","pdf":"document","contract":"document",
    # regulated domains
    "student":"education","education":"education","student record":"education","ferpa":"education",
    "child":"children","children":"children","minor":"children","coppa":"children",
    "trade secret":"trade_secret","confidential business information":"trade_secret","cbi":"trade_secret"
})

# ===== Additions to CANON["context"] =====
CANON["context"].update({
    # SDLC
    "development":"dev","dev":"dev","developer environment":"dev","local dev":"dev",
    "continuous integration":"ci","ci":"ci","build":"ci","pipeline":"ci",
    "quality assurance":"qa","qa":"qa","testing env":"qa",
    # inference
    "inference":"inference","serving":"inference","online serving":"inference","runtime inference":"inference",
    # surfaces
    "on-prem":"onprem","on prem":"onprem","onprem":"onprem",
    "cloud":"cloud","saas":"cloud","hosted":"cloud",
    "edge":"edge","on-device":"edge","on device":"edge",
    # modes
    "batch":"batch","offline":"batch",
    "stream":"stream","streaming":"stream","realtime":"stream","real-time":"stream",
    # regulatory designations
    "regulated":"regulated","compliance scope":"regulated","regulated scope":"regulated"
})


import unicodedata
import re

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _tok_norm(s: str) -> str:
    s = _strip_accents(s or "")
    s = s.lower().strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def _singularize(s: str) -> str:
    # very light singularization (good enough for vocab)
    if s.endswith("ies"): return s[:-3] + "y"
    if s.endswith("sses"): return s[:-2]
    if s.endswith("ses"): return s[:-2]
    if s.endswith("s") and not s.endswith("ss"): return s[:-1]
    return s

def _canonicalize_one(v: str, enum_set: set[str], syn_map: dict, fallback_keywords: dict[str, str]) -> str | None:
    raw = (v or "").strip()
    if not raw: return None
    n = _tok_norm(raw)
    # exact matches first
    if n in syn_map: 
        candidate = syn_map[n]
        return candidate if candidate in enum_set else None
    if n.upper() in {"PHI","PII"} and n.upper() in enum_set:
        return n.upper()
    if n in enum_set:
        return n
    # try singular form
    ns = _singularize(n)
    if ns in syn_map:
        candidate = syn_map[ns]
        return candidate if candidate in enum_set else None
    if ns in enum_set:
        return ns
    # keyword containment (safe, directional)
    for kw, canon in fallback_keywords.items():
        if kw in n:
            if canon in enum_set:
                return canon
    return None

def _canonicalize_list(vals, enum_set, syn_map, fallback_keywords):
    out=[]
    for v in (vals or []):
        c = _canonicalize_one(v, enum_set, syn_map, fallback_keywords)
        if c: out.append(c)
    # dedupe in order
    seen=set(); canon=[]
    for c in out:
        if c not in seen:
            seen.add(c); canon.append(c)
    return canon

# Keyword fallbacks to catch common phrases not in explicit syn maps
FALLBACK = {
  "actor": {
    "governance": "auditor",
    "security": "security",
    "privacy": "privacy",
    "moderation": "admin",
    "review": "evaluator",   # "human review"
    "ops ": "provider",
    " operations": "provider"
  },
  "data_domain": {
    "biometric": "biometric",
    "genomic": "health",
    "clinical": "health",
    "medical": "health",
    "credit card": "PII",
    "cardholder": "PII",
    "passport": "PII",
    "driver": "PII",
    "contact": "PII",
    "identifier": "PII",
    "ip ": "PII",
    "device id": "PII",
    "telemetry": "general",
    "log": "general",
    "snippet": "code",
    "repo": "code"
  },
  "context": {
    "prod": "prod",
    "runtime": "prod",
    "customer": "prod",
    "preprod": "staging",
    "canary": "staging",
    "uat": "staging",
    "experiment": "eval",
    "ab test": "eval",
    "a/b": "eval",
    "multi tenant": "tenant",
    "workspace": "tenant",
    "git": "repo",
    "pull request": "repo",
    "merge request": "repo",
    "regulated": "high-risk",
    "safety": "high-risk",
    "external": "public",
    "intranet": "internal",
    "employee": "internal",
    "research": "research",
    "train": "training",
    "fine": "finetune"
  }
}

def canonicalize_scope(rule: dict) -> None:
    sc = rule.setdefault("scope", {})
    sc["actor"] = _canonicalize_list(sc.get("actor", []), ACTOR_ENUM, CANON["actor"], FALLBACK["actor"])
    sc["data_domain"] = _canonicalize_list(sc.get("data_domain", []), DOMAIN_ENUM, CANON["data_domain"], FALLBACK["data_domain"])
    sc["context"] = _canonicalize_list(sc.get("context", []), CONTEXT_ENUM, CANON["context"], FALLBACK["context"])


# ------------------------ JSON Schema ------------------------
RULE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rule_id", "source", "scope", "hazard", "conditions",
                 "exceptions", "requirements", "evidence", "severity"],
    "properties": {
        "rule_id": {"type": ["string", "null"]},
        "source": {
            "type": "object",
            "required": ["doc", "citation", "span_id"],
            "additionalProperties": False,
            "properties": {
                "doc": {"type": "string"},
                "citation": {"type": "string"},
                "span_id": {"type": "string"}
            }
        },
        "scope": {
            "type": "object",
            "required": ["actor", "data_domain", "context"],
            "additionalProperties": False,
            "properties": {
            "actor": {"type": "array", "items": {"type": "string", "enum": sorted(list(ACTOR_ENUM))}},
            "data_domain": {"type": "array", "items": {"type": "string", "enum": sorted(list(DOMAIN_ENUM))}},
            "context": {"type": "array", "items": {"type": "string", "enum": sorted(list(CONTEXT_ENUM))}}
            }
        },
        "hazard": {"type": "string"},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "exceptions": {"type": "array", "items": {"type": "string"}},
        "requirements": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
    }
}
ARRAY_OF_RULES_SCHEMA = {"type": "array", "items": RULE_SCHEMA, "minItems": 0}
RULES_VALIDATOR = Draft202012Validator(ARRAY_OF_RULES_SCHEMA)
RULES_ENVELOPE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rules"],
    "properties": {
        "rules": ARRAY_OF_RULES_SCHEMA
    }
}

# ------------------------ Few-shot library ------------------------
FEWSHOTS: Dict[str, List[Dict[str, Any]]] = {
    "hipaa": [
        {
            "chunk": "A covered entity may not use or disclose protected health information except as permitted or required by this subpart.",
            "source": {"doc":"HIPAA Privacy Rule","citation":"45 CFR §164.502(a)"},
            "rules": [
                {
                    "rule_id": None,
                    "source": {"doc":"HIPAA Privacy Rule","citation":"45 CFR §164.502(a)","span_id":"${SPAN_ID}"},
                    "scope": {"actor":["model","agent"],"data_domain":["PHI"],"context":["prod","tenant"]},
                    "hazard":"Unauthorized PHI disclosure",
                    "conditions":["requesting party is not authorized","tenant_id_out != tenant_id_in"],
                    "exceptions":["treatment, payment, or healthcare operations","valid patient authorization"],
                    "requirements":["do not disclose PHI outside permitted purposes or tenant boundary"],
                    "evidence":["tenant identifiers","redaction log spans"],
                    "severity":"high"
                }
            ]
        }
    ],
    "who": [
        {
            "chunk": "Developers should ensure clinical guidance generated by AI is supported by high-quality evidence and includes citations where appropriate.",
            "source":{"doc":"WHO Guidance on AI for Health","citation":"Recommendations"},
            "rules":[
                {
                    "rule_id": None,
                    "source": {"doc":"WHO Guidance on AI for Health","citation":"Recommendations","span_id":"${SPAN_ID}"},
                    "scope": {"actor":["model","agent"],"data_domain":["health"],"context":["prod","eval"]},
                    "hazard":"Unsupported medical guidance",
                    "conditions":["response contains diagnosis or treatment claims"],
                    "exceptions":["explicit non-clinical info with referral to professional care"],
                    "requirements":["include credible citations or refuse safely"],
                    "evidence":["citation URLs or DOIs","verifier score"],
                    "severity":"high"
                }
            ]
        }
    ],
    "euai": [
        {
            "chunk": "High-risk AI systems shall be designed and developed with appropriate logging capabilities to ensure traceability of their functioning.",
            "source":{"doc":"EU AI Act 2024","citation":"Art. 12 Logging"},
            "rules":[
                {
                    "rule_id": None,
                    "source":{"doc":"EU AI Act 2024","citation":"Art. 12 Logging","span_id":"${SPAN_ID}"},
                    "scope":{"actor":["org"],"data_domain":["general"],"context":["prod","high-risk"]},
                    "hazard":"Missing auditability",
                    "conditions":["system is classified as high-risk"],
                    "exceptions":[],
                    "requirements":["retain logs linking events to model inputs/outputs and user context"],
                    "evidence":["log record IDs","timestamps","user context"],
                    "severity":"high"
                }
            ]
        }
    ],
    "ccby": [
        {
            "chunk": "You must give appropriate credit, provide a link to the license, and indicate if changes were made.",
            "source":{"doc":"CC BY 4.0","citation":"Sec. 3(a) Attribution"},
            "rules":[
                {
                    "rule_id": None,
                    "source":{"doc":"CC BY 4.0","citation":"Sec. 3(a) Attribution","span_id":"${SPAN_ID}"},
                    "scope":{"actor":["model","agent"],"data_domain":["code","text"],"context":["repo","prod"]},
                    "hazard":"Missing required attribution",
                    "conditions":["output includes material from CC BY 4.0 licensed source"],
                    "exceptions":["purely factual data without creative expression"],
                    "requirements":["include attribution and license link or avoid verbatim copying"],
                    "evidence":["attribution block","license URL"],
                    "severity":"medium"
                }
            ]
        }
    ]
}
def guess_corpus_name(doc: str) -> str:
    d = (doc or "").lower()
    if "hipaa" in d: return "hipaa"
    if "who" in d and "health" in d: return "who"
    if "eu ai act" in d or ("art." in d and "eu" in d): return "euai"
    if "cc by" in d or "creative commons" in d: return "ccby"
    return "generic"

# ------------------------ OpenAI client & model selection ------------------------
# ------------------------ Provider selection (OpenAI / Gemini) ------------------------
PROVIDERS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,  # default OpenAI
        "family_hint": ("gpt-4o", "gpt-4", "gpt-5"),  # only a hint; we still list models
    },
    "gemini": {
        "env_key": "GOOGLE_API_KEY",  # from Google AI Studio
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "family_hint": ("gemini",),   # we will filter discovered models with this
    },
    # (optional) if you later want Vertex AI’s compatibility endpoint, add another entry here
}

def _get_client_for_provider(provider: str):
    """
    Returns an OpenAI client configured either for OpenAI (default) or Google's
    OpenAI-compatible Gemini endpoint. Requires the right API key in the right env var.
    """
    provider = (provider or "openai").lower()
    cfg = PROVIDERS.get(provider)
    if not cfg:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {sorted(PROVIDERS)}")

    key = os.getenv(cfg["env_key"], "")
    if not key:
        raise RuntimeError(f"{cfg['env_key']} not set in environment for provider '{provider}'.")

    from openai import OpenAI
    if cfg["base_url"]:
        return OpenAI(api_key=key, base_url=cfg["base_url"])
    return OpenAI(api_key=key)

# Remove/replace the old PREFERRED_MODELS
MODEL_FALLBACKS = {
    "openai": [
        # just preferences; we will **also** list models dynamically:
        "gpt-5-thinking", "gpt-5", "gpt-5-mini" , "gpt-4.1", "gpt-4o", "gpt-4o-mini"
    ],
    "gemini": [
        # preferences based on current docs; still discovered dynamically:
        "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"
    ],
}

def list_available_models(client) -> list[str]:
    """Return model IDs from the provider using the OpenAI-compatible `models.list()`."""
    try:
        return [m.id for m in client.models.list().data]
    except Exception as e:
        # best effort; caller will do probe fallback
        print(f"Warning: listing models failed: {e}")
        return []

def select_working_model(client, provider: str, user_model: Optional[str]=None) -> str:
    """
    Strategy:
    1) If user_model supplied, try it directly.
    2) Else list models from provider, prefer families that look relevant.
    3) If still unsure, probe a small completion with a short preference list per provider.
    """
    if user_model:
        try:
            # quick sanity ping
            _ = client.chat.completions.create(model=user_model, messages=[{"role":"user","content":"[1]"}])
            return user_model
        except Exception:
            print(f"Note: user-supplied model '{user_model}' failed probe; falling back…")

    discovered = list_available_models(client)
    fam_hint = tuple(PROVIDERS.get(provider, {}).get("family_hint", ()))
    candidates = []

    # Prefer discovered models that match the provider's family hint
    if discovered:
        if fam_hint:
            for m in discovered:
                low = (m or "").lower()
                if any(h in low for h in fam_hint):
                    candidates.append(m)
        # if hint produced nothing, keep all discovered
        if not candidates:
            candidates = discovered[:]

    # Merge in static fallbacks (keeping order & uniqueness)
    for m in MODEL_FALLBACKS.get(provider, []):
        if m not in candidates:
            candidates.append(m)

    # Probe in order
    probe = [{"role": "user", "content": "[1]"}]
    for m in candidates:
        try:
            r = client.chat.completions.create(model=m, messages=probe)
            if r and getattr(r, "choices", None):
                return m
        except Exception:
            continue

    # Absolute last resort: something cheap-ish on each side
    return ("gpt-4o-mini" if provider == "openai" else "gemini-2.0-flash")


# ------------------------ Prompt builder ------------------------
def _fmt_sorted(s): return ", ".join(sorted(s))

def build_system_msg() -> str:
    return (
        "ROLE: Compliance rule extractor.\n"
        "TASK: Convert the provided policy clause (single sentence only) into one or more ATOMIC, OPERATIONALLY TESTABLE rules.\n"
        'OUTPUT: JSON OBJECT ONLY with a single key "rules" (array). No prose, no markdown, no comments.\n\n'

        "HARD CONSTRAINTS (read carefully):\n"
        "• NO NEW FACTS: Do not introduce actors, artifacts, metrics, filenames, IDs, URLs, or citations not present in the clause.\n"
        "• NO EVIDENCE INVENTION: If the clause names no artifact type, set evidence=[]. Never fabricate filenames/paths/IDs.\n"
        "• CONCRETE HAZARD ONLY: One specific risk plainly stated or unambiguously implied by the clause (e.g., “unauthorized PHI disclosure”, “action based on unverified identification”). No vague labels like “ethics”.\n"
        "• TRACEABILITY (STRICT): Copy source.doc, source.citation, and source.span_id EXACTLY from the input. If any is missing → {\"rules\":[]}.\n"
        "• PROHIBITIONS REQUIRE CONDITIONS: Any prohibition must include ≥1 concrete trigger in conditions[].\n"
        "• CANONICAL SCOPE ONLY: Use ONLY canonical tokens (from the provided enums). Prefer explicit grounding (actor text appears in the clause). If no actor text appears, you MAY use Controlled Inference (below) to infer exactly one actor that maps to the actor enum; if no cue applies, output {\"rules\":[]}.\n"
        "• ORG-FACING FILTER: Emit rules ONLY if the duty is on implementers/users (developer, provider, deployer, org, user, admin, security, privacy, risk, data_steward, maintainer, evaluator, analyst, pm, support).\n"
        "• REGULATOR-ONLY DUTIES: If the only actors in the clause are regulators/public authorities (e.g., “the Commission”, “national competent authority”, “notified body”, “supervisory authority”) → {\"rules\":[]}. Do NOT transpose to org/provider/deployer.\n"
        "• ACTOR TRANSPOSITION BAN: Never replace a regulator/government actor with an implementer/user actor.\n"
        "• PASSIVE/ACTORLESS SENTENCES: Allowed if the actor can be derived via Controlled Inference; otherwise → {\"rules\":[]}.\n"
        "• ANNEX/ARTICLE NARROWING: If the clause narrows scope via Annex/Article/Section/Clause/Appendix, include a short trigger in conditions[] (e.g., “Annex III 1(a) system”).\n"
        "• “NO … UNLESS …” MAPPING: Phrases of the form “no [action] … unless [condition]” MUST be encoded as a prohibition; put the “unless …” part into conditions[].\n"
        "• CARVE-OUT SENTENCES: A clause like “The requirement shall not apply …” with no canonical actor does NOT form a new rule. Attach as exceptions[] ONLY if the same sentence also states the base duty (otherwise drop). If not applicable, drop.\n"
        "• EARLY EXIT: If you cannot meet ATOMICITY + CONCRETE HAZARD + at least one actionable requirement tied to an actor (explicit or inferred per Controlled Inference), output {\"rules\":[]}.\n\n"

        "SCHEMA (structure only):\n"
        "{ \"rules\": [\n"
        "  {\n"
        "    \"rule_id\": null,\n"
        "    \"source\": {\"doc\": \"\", \"citation\": \"\", \"span_id\": \"\"},\n"
        "    \"scope\": {\"actor\": [], \"data_domain\": [], \"context\": []},\n"
        "    \"hazard\": \"\",\n"
        "    \"conditions\": [],\n"
        "    \"exceptions\": [],\n"
        "    \"requirements\": [],\n"
        "    \"evidence\": [],\n"
        "    \"severity\": \"\"\n"
        "  }\n"
        "]}\n\n"

        "FIELD-BY-FIELD RULES:\n"
        "1) ATOMICITY: Split composite obligations so each rule contains exactly one obligation/prohibition. Prefer multiple short rules over a long one.\n"
        "2) DEONTIC MAPPING: shall/must/require → requirement; may not/shall not/prohibit → prohibition (+conditions[]). Also map “no [action] … unless [condition]” to a prohibition with the [condition] in conditions[].\n"
        "3) EXCEPTIONS: Move “unless/except/if not” fragments into exceptions[] as short, testable phrases.\n"
        "4) SCOPE ENUMS (canonical only; normalize synonyms that APPEAR IN CLAUSE when present):\n"
        f"   • scope.actor ∈ {{{_fmt_sorted(ACTOR_ENUM)}}}\n"
        f"   • scope.data_domain ∈ {{{_fmt_sorted(DOMAIN_ENUM)}}}\n"
        f"   • scope.context ∈ {{{_fmt_sorted(CONTEXT_ENUM)}}}\n"
        "   If the clause does not clearly support a value, leave []. Do not guess.\n"
        "5) HAZARD: One short, concrete risk label supported by the clause.\n"
        "6) CONDITIONS: Short, verifiable triggers stated in the clause (e.g., “before deployment”, “based on the system's identification”, “Annex III 1(a) system”).\n"
        "7) REQUIREMENTS (most important):\n"
        "   • Each requirement MUST be a verifiable action, written as an obligation of a canonical actor (explicit or inferred via Controlled Inference).\n"
        "   • The action must be checkable via outputs/config/process/record (e.g., “publish model card including known limitations”, “conduct bias analysis across protected subgroups”).\n"
        "   • Do NOT reference specific artifacts unless the clause names an artifact type (use generic labels only).\n"
        "   • 1-3 requirements per rule; each requirement is a single sentence.\n"
        "8) EVIDENCE:\n"
        "   • Include evidence ONLY if the clause names an artifact type (e.g., “impact assessment”, “risk management plan”, “audit report”, “model card”).\n"
        "   • Use generic artifact labels (e.g., “impact assessment (subgroup metrics)”); otherwise evidence=[].\n"
        "9) SEVERITY: One of {low, medium, high, critical}. Use “high” for regulatory/safety-impacting requirements; use “critical” only if violation directly endangers patient safety.\n"
        "10) QUOTING: You may include a short clause fragment in a condition/requirement in quotes to anchor the test, but keep fields succinct.\n"
        "11) COUNT & ORDER: Emit at most 6 rules per clause. Order by operational priority (safety > privacy > transparency > administrative).\n"
        "12) OUTPUT FORMAT: Return EXACTLY one JSON object with key \"rules\". No surrounding text, no code fences, no trailing commas.\n\n"

        "Controlled Inference (when no actor text appears): If the clause is imperative, infer EXACTLY ONE actor from the cues below, mapping to the actor enum. If multiple are plausible, choose \"org\"; never infer regulators/authorities.\n"
        "Cues → enum:\n"
        "- Product/Customer docs → org (use provider only if “service provider/platform” is named)\n"
        "- Build/train/design/evaluate/test → developer (if explicitly raters/testers → evaluator)\n"
        "- Deploy/release/operate/monitor → deployer (or admin when about admin/policy ops)\n"
        "- Report/notify/submit to oversight body → org\n"
        "- Artifact duties (impact assessment, transparency note, responsible release criteria/plan, evaluation plan, release plan, risk register) → org\n\n"

        "FINAL VALIDATION CHECK (MUST PASS FOR EVERY RULE):\n"
        "• source.doc/citation/span_id are exact copies from input;\n"
        "• scope uses ONLY canonical tokens (enums); actor is either (a) explicitly present in the clause, or (b) inferred via Controlled Inference; never infer regulators/authorities.\n"
        "• hazard is concrete and supported by the clause;\n"
        "• at least one actionable, checkable requirement is tied to that actor;\n"
        "• any prohibition includes ≥1 concrete condition;\n"
        "• evidence[] is empty unless an artifact type is named in the clause; no fabricated filenames/paths/IDs;\n"
        "• arrays are [] when truly not applicable.\n"
        "If ANY check fails for a rule, DROP that rule. If all candidate rules are dropped, output {\"rules\": []}.\n"
    )



def build_user_msg(chunk_text: str, doc: str, citation: str, span_id: str, corpus: str) -> List[Dict[str, Any]]:
    shots = FEWSHOTS.get(corpus, [])
    messages: List[Dict[str, Any]] = []
    # Few-shots
    for sh in shots[:2]:
        messages.append({"role":"user", "content": f"Text:\n{sh['chunk']}\n\nReturn JSON object with key 'rules'."})
        sh_rules_obj = {"rules": sh["rules"]}
        sh_rules = json.dumps(sh_rules_obj, ensure_ascii=False).replace("${SPAN_ID}", span_id)
        messages.append({"role":"assistant","content": sh_rules})
    # Actual extraction
    schema_str = json.dumps({
        "rules": [{
            "rule_id": None,
            "source": {"doc": "", "citation": "", "span_id": ""},
            "scope": {"actor": [], "data_domain": [], "context": []},
            "hazard": "", "conditions": [], "exceptions": [],
            "requirements": [], "evidence": [], "severity": ""
        }]
    }, ensure_ascii=False)

    messages.append({"role":"user","content": (
        "Extract rules as a JSON OBJECT ONLY of the form {\"rules\": [...]}. "
        "Follow this JSON schema exactly.\n"
        f"Schema example (structure only):\n{schema_str}\n\n"
        f"Document: {doc}\nCitation: {citation}\nSpan ID: {span_id}\n\nText:\n{chunk_text}\n"
        "Instructions:\n"
        "- Split composite requirements into separate rules.\n"
        "- Use canonical scope tokens (enums). If no actor text appears, you MAY infer exactly one actor using the Controlled Inference cues (org/developer/deployer/evaluator/admin/provider as applicable). If no cue applies, return {\"rules\":[]}.\n"
        "- Emit rules ONLY for mandatory obligations/prohibitions. Clauses that are permissive (e.g., 'may', 'shall have a choice of', 'as appropriate' without an objective trigger) must produce {\"rules\":[]}.\n"
        "- severity ∈ {low, medium, high, critical}.\n"
        "- No prose, no markdown — return the object only."
    )})
    return messages

# ------------------------ Error helpers ------------------------
def _print_openai_error(e):
    msg = getattr(e, "message", None) or str(e)
    try:
        body = getattr(e, "response", None)
        if body is not None and hasattr(body, "json"):
            try:
                j = body.json()
                msg += f"\nServer said: {json.dumps(j, ensure_ascii=False)}"
            except Exception:
                pass
    except Exception:
        pass
    print(f"OpenAI error: {msg}", file=sys.stderr)

# ------------------------ LLM call (schema/object/plain fallbacks) ------------------------
def call_llm_extract_hardened(client, model: str, messages: List[Dict[str, Any]], json_schema: Dict[str, Any]) -> str:
    """
    Robust Chat Completions caller across providers/models:

    Strategy:
      A) response_format=json_schema (strict object root)
      B) response_format=json_object (no schema)
      C) plain JSON (no response_format)

    Quirk handling:
      - Do NOT send 'temperature' (some models only accept default).
      - Prefer 'max_completion_tokens'. If provider rejects it, retry with 'max_tokens'.
    """
    sys_msg = {"role": "system", "content": build_system_msg()}
    full_msgs = [sys_msg] + messages

    def _chat(payload):
        # Normalize tokens: prefer max_completion_tokens; do not send max_tokens first.
        mt = payload.pop("max_completion_tokens", None) or payload.pop("max_tokens", None) or 10000
        payload["max_completion_tokens"] = mt

        # Never send temperature unless explicitly needed (many models reject overrides).
        payload.pop("temperature", None)

        try:
            resp = client.chat.completions.create(**payload)
            return resp.choices[0].message.content or ""
        except Exception as e:
            # Retry for common provider/model quirks.
            msg = getattr(e, "message", None) or str(e)

            # Some models do not support max_completion_tokens; retry with max_tokens.
            if ("Unsupported parameter" in msg and "max_completion_tokens" in msg) or ("'max_completion_tokens'" in msg):
                payload.pop("max_completion_tokens", None)
                payload["max_tokens"] = mt
                payload.pop("temperature", None)
                resp = client.chat.completions.create(**payload)
                return resp.choices[0].message.content or ""

            # Some models reject any explicit temperature value.
            if ("Unsupported value" in msg and "temperature" in msg) or ("param" in msg and "temperature" in msg):
                payload.pop("temperature", None)
                resp = client.chat.completions.create(**payload)
                return resp.choices[0].message.content or ""

            # Bubble up anything else.
            raise

    # ---------- Attempt A: structured json_schema ----------
    try:
        return _chat({
            "model": model,
            "messages": full_msgs,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "rules_object", "schema": json_schema, "strict": True}
            },
            "max_completion_tokens": 10000,
        })
    except Exception as e:
        _print_openai_error(e)

    # ---------- Attempt B: json_object wrapper ----------
    wrapper = {"role": "user", "content": 'Return ONLY: {"rules": <ARRAY>} where <ARRAY> follows the schema.'}
    try:
        return _chat({
            "model": model,
            "messages": full_msgs + [wrapper],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 10000,
        })
    except Exception as e:
        _print_openai_error(e)

    # ---------- Attempt C: plain JSON (no enforced format) ----------
    return _chat({
        "model": model,
        "messages": full_msgs,
        "max_completion_tokens": 10000,
    })


# ------------------------ Parsing & validation ------------------------
def parse_rules_text(raw_text: str) -> List[dict]:
    raw = (raw_text or "").strip()
    # Try dict wrapper: {"rules": [...]}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "rules" in obj and isinstance(obj["rules"], list):
            return obj["rules"]
    except Exception:
        pass
    # Try full array
    m = re.search(r"\[\s*\{.*\}\s*\]\s*$", raw, flags=re.S)
    if m:
        return json.loads(m.group(0))
    # First array anywhere
    m = re.search(r"\[\s*\{.*?\}\s*\]", raw, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("No JSON array found in model output.")


# --- Heuristic severity classifier ---
REGULATORY_HINTS = ("hipaa", "eu ai act", "gdpr", "45 cfr", "regulation", "law", "statute")
GUIDANCE_HINTS   = ("who", "nist", "oecd", "standard", "guidance", "recommendation", "framework")
ESCALATE_HARD    = ("criminal", "sanction", "liable", "penalty", "fines", "breach")
CRITICAL_HINTS   = ("safety-critical", "life-critical", "high-risk", "systemic risk")

def _severity_from_context(doc:str, citation:str, text_blob:str) -> str:
    d = (doc or "").lower()
    c = (citation or "").lower()
    t = (text_blob or "").lower()

    if any(k in d or k in c or k in t for k in CRITICAL_HINTS):
        return "critical"
    if any(k in d or k in c for k in REGULATORY_HINTS):
        return "high"
    if any(k in d or k in c for k in GUIDANCE_HINTS):
        return "medium"
    if any(k in t for k in ESCALATE_HARD):
        return "high"
    return "medium"

def explode_composite_rules(rules: List[dict]) -> List[dict]:
    out=[]
    for r in rules:
        reqs = r.get("requirements", []) or [""]
        if len(reqs) <= 1:
            out.append(r); continue
        # split: one rule per requirement
        for i, req in enumerate(reqs, 1):
            r2 = json.loads(json.dumps(r))  # deep copy
            r2["requirements"] = [req]
            # ensure distinct ids later; leave rule_id None so the pipeline assigns a unique one
            r2["rule_id"] = None
            out.append(r2)
    return out

def _is_prohibition(requirements: List[str]) -> bool:
    text = " ".join(requirements).lower()
    return any(kw in text for kw in ("must not", "may not", "shall not", "prohibit", "forbid", "forbidden"))

def _has_conditions(conditions: List[str]) -> bool:
    return any(s.strip() for s in (conditions or []))

def _normalize_phrase(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+"," ", s.lower()).strip()
    s = re.sub(r"\s+"," ", s)
    return s

def verify_rule_semantics(rule: dict) -> List[str]:
    probs=[]
    # Consistency: prohibitions need conditions
    if _is_prohibition(rule.get("requirements", [])) and not _has_conditions(rule.get("conditions", [])):
        probs.append("Prohibition without conditions[]")
    # Conflict: same predicate appears in requirements and exceptions
    req_norm = {_normalize_phrase(x) for x in rule.get("requirements", [])}
    exc_norm = {_normalize_phrase(x) for x in rule.get("exceptions", [])}
    overlap = req_norm.intersection(exc_norm)
    if overlap:
        probs.append(f"Requirement/exception conflict on: {sorted(list(overlap))[:2]}")
    # Minimality: hazard present
    if not (rule.get("hazard","").strip()):
        probs.append("Missing hazard")
    return probs


def normalize_and_validate_rules(arr: List[dict], defaults: Dict[str, str], validator: Draft202012Validator) -> List[dict]:
    for r in arr:
        # list fields: coerce to strings & drop empties
        for k in ["conditions", "exceptions", "requirements", "evidence"]:
            r.setdefault(k, [])
            r[k] = [str(x).strip() for x in r[k] if str(x).strip()]

        r.setdefault("rule_id", None)

        src = r.setdefault("source", {})
        src.setdefault("doc", defaults.get("doc", ""))
        src.setdefault("citation", defaults.get("citation", ""))
        src.setdefault("span_id", defaults.get("span_id", ""))

        sc = r.setdefault("scope", {})
        sc.setdefault("actor", [])
        sc.setdefault("data_domain", [])
        sc.setdefault("context", [])

        # severity: prefer model value if valid, else infer
        sev = (r.get("severity") or "").lower()
        if sev not in ["low", "medium", "high", "critical"]:
            r["severity"] = _severity_from_context(
                src.get("doc", ""), src.get("citation", ""),
                " ".join([r.get("hazard", "")] + r.get("requirements", []))
            )

        canonicalize_scope(r)

    # validate
    errors = sorted(validator.iter_errors(arr), key=lambda e: e.path)
    if errors:
        msgs = [f"path={list(e.path)}: {e.message}" for e in errors]
        raise ValidationError("Schema validation failed: " + " | ".join(msgs))
    return arr

# =========================
# New: Grammar decoder (stub) + Judge/Repair + SMT + Evidence gate + Counterfactuals
# =========================

def decode_rules(client, model: str, messages: List[Dict[str, Any]], *, decoder: str="jsonschema") -> List[dict]:
    """
    decoder: "jsonschema" (default) or "grammar".
    The 'grammar' path still uses the same API but adds a preface that mimics a PEG AST;
    keeps compatibility without extra libs.
    """
    if decoder == "grammar":
        # Light-weight grammar nudge: add a short AST instruction before the existing messages.
        grammar_hint = {
            "role": "system",
            "content": "Use a minimal AST-style structure internally; still output a JSON object {\"rules\": [...]} that conforms exactly to the provided schema."
        }
        messages = [grammar_hint] + messages
    raw = call_llm_extract_hardened(client, model, messages, RULES_ENVELOPE_SCHEMA)
    arr = parse_rules_text(raw)
    return arr




def evidence_gate_ok(rule: dict, *, require_span: bool=True, allow_domains: Optional[List[str]]=None) -> List[str]:
    problems=[]
    src = rule.get("source", {}) or {}

    if require_span and not (src.get("span_id") or "").strip():
        problems.append("Missing source.span_id")
    # Ensure we have a clean, string-only list
    ev_vals = rule.get("evidence", []) or []
    ev_norm = [str(x).strip() for x in ev_vals if x is not None and str(x).strip()]

    if not ev_norm:
        problems.append("Missing evidence[]")

    if allow_domains:
        allow = [d.lower().lstrip(".") for d in allow_domains]
        for e in ev_norm:
            el = e.lower()
            if "http" not in el:
                continue
            host_key = _host_etld1(e)
            if not host_key:
                problems.append(f"Evidence URL has no valid hostname: {e}")
                continue
            if not any(host_key == ad or host_key.endswith("." + ad) or host_key.endswith(ad) for ad in allow):
                problems.append(f"Evidence URL host not allowed: {e}")

    return problems

def judge_rule(rule: dict, *, allow_domains: Optional[List[str]]=None, use_evidence_gate: bool=True) -> List[str]:
    issues = verify_rule_semantics(rule)
    if use_evidence_gate:
        # issues += evidence_gate_ok(rule, allow_domains=allow_domains)
        print("Skipping evidence gate")
    # minimal scope sanity
    sc = rule.get("scope", {})
    if not any(sc.get(k) for k in ("actor","data_domain","context")):
        issues.append("Missing scope fields")
    return issues

import re
from urllib.parse import urlparse

# Canonical enums (same as your extractor schema)
_SCOPE_ACTOR = {"admin","agent","analyst","auditor","data_owner","data_steward","deployer","developer","evaluator","legal","maintainer","model","org","pm","privacy","provider","risk","security","support","user"}
_SCOPE_DD    = {"PHI","PII","audio","biometric","children","code","credentials","document","education","financial","general","geolocation","health","image","network","secrets","telemetry","text","trade_secret","video"}
_SCOPE_CTX   = {"batch","ci","cloud","dev","edge","eval","finetune","high-risk","inference","internal","onprem","prod","public","qa","regulated","repo","research","staging","stream","tenant","training"}
_SEVERITY    = {"low","medium","high","critical"}

_ALLOWED_TOP_KEYS = {"rule_id","source","scope","hazard","conditions","exceptions","requirements","evidence","severity"}
_ALLOWED_SOURCE   = {"doc","citation","span_id"}
_ALLOWED_SCOPE    = {"actor","data_domain","context"}

def _canon_token(tok: str, space: set[str]) -> str | None:
    t = (tok or "").strip()
    if not t:
        return None
    # exact hit
    if t in space:
        return t
    # basic normalization
    low = t.lower().replace(" ", "").replace("-", "")
    for s in space:
        if low == s.lower().replace(" ", "").replace("-", ""):
            return s
    return None

def _prune_unknowns(rule: dict) -> dict:
    # keep only allowed top-level keys
    rule = {k: v for k, v in rule.items() if k in _ALLOWED_TOP_KEYS}
    # source
    src = rule.get("source", {})
    if not isinstance(src, dict): src = {}
    rule["source"] = {k: src.get(k, "") for k in _ALLOWED_SOURCE}
    # scope
    sc = rule.get("scope", {})
    if not isinstance(sc, dict): sc = {}
    sc2 = {k: sc.get(k, []) for k in _ALLOWED_SCOPE}
    rule["scope"] = sc2
    # arrays normalize
    for k in ("actor","data_domain","context"):
        arr = rule["scope"].get(k, [])
        if not isinstance(arr, list): arr = [arr] if arr else []
        # dedupe/preserve order
        seen = set(); out = []
        for x in arr:
            x = x if isinstance(x, str) else str(x)
            if x not in seen:
                seen.add(x); out.append(x)
        rule["scope"][k] = out
    for k in ("conditions","exceptions","requirements","evidence"):
        arr = rule.get(k, [])
        if not isinstance(arr, list): arr = [arr] if arr else []
        out, seen = [], set()
        for x in arr:
            s = (x if isinstance(x, str) else str(x)).strip()
            if s and s not in seen:
                seen.add(s); out.append(s)
        rule[k] = out
    # strings
    rule["hazard"]   = (rule.get("hazard") or "").strip()
    sev = (rule.get("severity") or "").strip().lower()
    if sev not in _SEVERITY:
        sev = "high" if rule["hazard"] else "medium"
    rule["severity"] = sev
    # rule_id may be null/str; leave as-is
    return rule

def _canon_scope(rule: dict) -> None:
    # canonicalize enums; drop non-canonical
    ac, dd, cx = [], [], []
    for t in rule["scope"].get("actor", []):
        c = _canon_token(t, _SCOPE_ACTOR); 
        if c: ac.append(c)
    for t in rule["scope"].get("data_domain", []):
        c = _canon_token(t, _SCOPE_DD); 
        if c: dd.append(c)
    for t in rule["scope"].get("context", []):
        c = _canon_token(t, _SCOPE_CTX); 
        if c: cx.append(c)
    # defaults if ended empty
    rule["scope"]["actor"] = ac or ["org"]
    rule["scope"]["data_domain"] = dd or ["general"]
    rule["scope"]["context"] = cx or ["internal"]

def _filter_evidence_domains(rule: dict, allow_domains: list[str] | None) -> None:
    if not allow_domains: 
        return
    allowed = tuple(d.lower().strip() for d in allow_domains)
    out = []
    for ev in rule.get("evidence", []):
        s = ev.strip()
        if not s:
            continue
        # keep file-like or plain artifact names
        if "://" not in s and not re.match(r"^www\.", s, re.I):
            out.append(s); continue
        try:
            netloc = urlparse(s).netloc.lower()
            if any(netloc.endswith(dom) for dom in allowed):
                out.append(s)
        except Exception:
            # if parse fails, keep original string (non-URL)
            out.append(s)
    rule["evidence"] = out

def _safe_json_object_completion(client, model: str, messages: list[dict], max_tokens: int = 1000):
    kwargs = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_tokens,
    }
    return client.chat.completions.create(**kwargs)

def repair_rule_llm(
    client,
    model: str,
    rule: dict,
    issues: list[str],
    original_text: str,
    *,
    allow_domains: list[str] | None = None,
    logger: "logging.Logger" = logger,
) -> dict:
    """
    Ask the model to perform minimal, local edits to fix issues; keep schema identical.
    - Forces JSON output
    - Retries once with a stricter instruction if schema breaks
    - Prunes/normalizes to schema after the call
    """
    def _prompt(minimal: bool = True) -> list[dict]:
        # System prompt aligned with extractor + judge, updated for Controlled Inference
        sys = (
            "ROLE: Compliance RULE REPAIR agent.\n"
            "GOAL: Apply MINIMAL, LOCAL edits to the provided JSON rule to fix the listed issues so it becomes "
            "ATOMIC and OPERATIONALLY TESTABLE—WITHOUT inventing facts beyond the source_text.\n\n"

            "STRICT CONSTRAINTS:\n"
            "• SOURCE-BOUND: Only use information plainly present in source_text. Do NOT invent actors, artifacts, "
            "  URLs, metrics, filenames, IDs, or hazards.\n"
            "• SCHEMA LOCK: Keep EXACT keys and types:\n"
            "  {rule_id, source{doc,citation,span_id}, scope{actor[],data_domain[],context[]}, "
            "   hazard, conditions[], exceptions[], requirements[], evidence[], severity}.\n"
            "  Do not add keys; do not rename keys.\n"
            "• CANONICAL SCOPE ONLY: Use ONLY canonical tokens (from the provided enums). Prefer explicit grounding "
            "  (actor text appears in the clause). If no actor text appears, you MAY use Controlled Inference (below) "
            "  to infer EXACTLY ONE actor that maps to the actor enum. If no cue applies, LEAVE actor EMPTY; do not invent.\n"
            "  If not clearly supported, leave subfields as []. Never synthesize non-canonical actors.\n"
            "• PROHIBITIONS REQUIRE CONDITIONS: If a requirement is a prohibition (e.g., 'must not', 'shall not', 'may not'), "
            "  add at least ONE concrete trigger in conditions[] ONLY IF such a trigger is stated in source_text "
            "  (e.g., 'before deployment', 'processing PHI', 'used for clinical decision support'). If no trigger exists, "
            "  DO NOT fabricate one—leave as-is and fix other issues you can.\n"
            "• HAZARD: Must be a single, concrete risk explicitly supported by source_text. Do NOT invent a hazard. If no "
            "  concrete hazard is present in source_text, keep hazard as '' (empty).\n"
            "• REQUIREMENTS: Keep only concrete, testable actions. If requirements[] is truly missing in source_text, leave it [].\n"
            "• EVIDENCE: Only include if source_text names an artifact type (e.g., 'impact assessment', 'model card'). Use generic "
            "  labels (e.g., 'impact assessment (subgroup metrics)'). Never add filenames or URLs. Otherwise evidence=[].\n"
            "• EXCEPTIONS: Keep short, testable phrases taken from source_text (e.g., 'unless regulator approval is obtained').\n"
            "• ATOMICITY: If multiple obligations are present inside one requirement, split them into separate requirement lines "
            "  within the same rule only if the text makes them separable without invention. Otherwise keep one.\n"
            "• SEVERITY: If invalid or missing, set based on context (regulatory/safety ⇒ 'high'; explicit safety-critical ⇒ 'critical'; "
            "  otherwise 'medium').\n\n"

            "WHAT TO FIX (tie directly to judge labels):\n"
            "1) Prohibition without concrete conditions → If prohibition detected and source_text contains a concrete trigger, "
            "   copy that trigger into conditions[]. Do not invent triggers.\n"
            "2) Missing hazard → If source_text contains a specific failure mode, set hazard to that concise phrase; else leave ''.\n"
            "3) Missing requirements → If source_text includes a concrete, testable action, add it succinctly; else leave [].\n"
            "4) Scope fields empty → If source_text clearly assigns an org-facing duty, map actors/contexts/domains to canonical "
            "   tokens; otherwise keep [].\n"
            "5) Conflicts between requirements and exceptions → Rewrite exceptions to match the clause text precisely or "
            "   prune contradictory text that isn’t supported by source_text. Do not change the meaning.\n\n"

            "CONTROLLED INFERENCE (actor cues when no actor text appears):\n"
            "- Product/Customer docs → org (use provider only if 'service provider/platform' is named)\n"
            "- Build/train/design/evaluate/test → developer (if explicitly raters/testers → evaluator)\n"
            "- Deploy/release/operate/monitor → deployer (or admin when about admin/policy ops)\n"
            "- Report/notify/submit to oversight body → org\n"
            "- Artifact duties (impact assessment, transparency note, responsible release criteria/plan, evaluation plan, "
            "  release plan, risk register) → org\n\n"

            "CANONICAL ENUMS (use ONLY these tokens; otherwise leave the subfield empty):\n"
            f"• actor ∈ {sorted(list(ACTOR_ENUM))}\n"
            f"• data_domain ∈ {sorted(list(DOMAIN_ENUM))}\n"
            f"• context ∈ {sorted(list(CONTEXT_ENUM))}\n\n"

            "OUTPUT: Return ONLY the corrected JSON object for the rule (no wrapper, no prose)."
        )
        if not minimal:
            # Even stricter reminder for the retry pass
            sys += (
                "\nHARD MODE REMINDER: Do NOT add or rename keys. Keep the exact schema structure and value types. "
                "If a field cannot be repaired without inventing content, leave it empty."
            )

        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps({
                "rule": rule,
                "issues": issues,
                "source_text": original_text
            }, ensure_ascii=False)}
        ]



    # 1) First attempt (minimal edits)
    try:
        logger.debug("[repair] request (minimal) → model=%s", model)
        resp = _safe_json_object_completion(client, model, _prompt(minimal=True), max_tokens=1500)
        raw = resp.choices[0].message.content or "{}"
        obj = json.loads(raw)
    except Exception as e:
        logger.warning("[repair] first attempt failed: %s", e)
        obj = {}

    # 2) If not a dict or schema looks off, try a stricter retry
    if not isinstance(obj, dict) or any(k not in _ALLOWED_TOP_KEYS for k in obj.keys()):
        try:
            logger.info("[repair] retry with STRICT schema guard")
            resp = _safe_json_object_completion(client, model, _prompt(minimal=False), max_tokens=1500)
            raw = resp.choices[0].message.content or "{}"
            obj = json.loads(raw)
        except Exception as e:
            logger.error("[repair] strict retry failed: %s", e)
            # Return the original rule if both attempts fail
            return rule

    # 3) Accept either {"rule": {...}} or raw rule
    fixed = obj.get("rule") if isinstance(obj, dict) and "rule" in obj else obj
    if not isinstance(fixed, dict):
        logger.warning("[repair] invalid JSON shape; returning original")
        return rule

    # 4) Sanitize/normalize to schema
    fixed = _prune_unknowns(fixed)
    _canon_scope(fixed)
    _filter_evidence_domains(fixed, allow_domains)

    # 5) Minimal hygiene: ensure non-empty hazard if requirements exist
    if not fixed.get("hazard", "").strip() and fixed.get("requirements"):
        fixed["hazard"] = "Unspecified hazard"

    # 6) Final log
    logger.debug("[repair] final rule: %s", json.dumps(fixed, ensure_ascii=False))

    return fixed

def smt_check_conflicts(new_rule: dict, existing_rules: List[dict]) -> List[str]:
    """
    Strong SMT check (if z3 present) for contradictory obligations on the same normalized predicate.
    We model a single 'world' with boolean variables for each canonical scope token:
      - Actors:    user, model, agent, org, provider, deployer, developer, evaluator, admin, auditor
      - Domains:   PHI, PII, health, code, text, image, audio, biometric, general
      - Contexts:  prod, staging, eval, tenant, repo, high-risk, public, internal, research, training, finetune

    For each rule r and each requirement clause req in r.requirements:
      - Let P = normalize(req)  (same key used elsewhere)
      - Let Polarity(P) ∈ {Require, Forbid}
      - Let Apply_r be the conjunction of ORs over the scope lists that are non-empty; empty lists are wildcards (True).
      - Conditions/exceptions are free text; we conservatively *allow* a world where conditions are satisfiable and exceptions do not hold.

    We assert:
      Apply_r  => State[P] == Polarity(r, req)

    A contradiction exists if, for some predicate P, there exists a world where:
      Apply_new  and  Apply_existing  and  Polarity(new,P) != Polarity(existing,P)
    Because State[P] is a single-valued enum, opposite implications render the formula UNSAT.
    We probe each such pair and report conflicts precisely.

    Returns: list of human-readable conflict strings. Empty if none or if z3 not available.
    """
    if not HAVE_Z3:
        return []  # Fallback: keep silent if z3 not available
    from z3 import Solver, Bool, EnumSort, Const, Or, And, Implies, BoolVal, sat

    # --- Helpers -------------------------------------------------------------
    def pred_key(req: str) -> str:
        return _normalize_phrase(req)

    # Build the universe of scope variables that actually appear.
    # Using the canonical enums ensures we don't miss tokens; it's fine to include all.
    actor_vars = {a: None for a in sorted(ACTOR_ENUM)}
    domain_vars = {d: None for d in sorted(DOMAIN_ENUM)}
    context_vars = {c: None for c in sorted(CONTEXT_ENUM)}

    # Create boolean variables for the "world"
    for a in actor_vars:
        actor_vars[a] = Bool(f"a_{a}")
    for d in domain_vars:
        domain_vars[d] = Bool(f"d_{d}")
    for c in context_vars:
        context_vars[c] = Bool(f"c_{c}")

    # At least one token in each dimension should hold in a world (prevents degenerate empty worlds)
    base_world_constraints = [
        Or(*actor_vars.values()),
        Or(*domain_vars.values()),
        Or(*context_vars.values())
    ]

    # Enum for predicate state: Unknown, Require, Forbid
    PredState, (Unknown, Require, Forbid) = _get_pred_enum()

    # We'll lazily create a State[P] const when first needed
    pred_state = {}  # key -> z3 Const

    def get_state_for(key: str):
        if key not in pred_state:
            pred_state[key] = Const(f"state__{key}", PredState)
        return pred_state[key]

    # Quick canonical OR over a list of tokens in a given dim; empty => True (wildcard)
    def dim_match(tokens: List[str], varmap: Dict[str, Any]) -> Any:
        toks = [t for t in (tokens or []) if t in varmap]
        if not toks:
            return BoolVal(True)
        return Or(*[varmap[t] for t in toks])

    # Build Apply_r for a rule over the world vars
    def apply_formula(rule: dict):
        sc = rule.get("scope", {}) or {}
        a = dim_match(sc.get("actor", []), actor_vars)
        d = dim_match(sc.get("data_domain", []), domain_vars)
        c = dim_match(sc.get("context", []), context_vars)
        # We treat conditions/exceptions conservatively: there exists a world where conditions hold and exceptions do not.
        # So we do NOT add extra constraints for them here (they'd just be extra booleans that can be set).
        return And(a, d, c)

    # Compute polarity for a *single requirement string*
    def req_polarity(req: str) -> str:
        t = req.lower()
        if any(w in t for w in ["must not", "shall not", "may not", "forbid", "forbidden", "prohibit"]):
            return "forbid"
        return "require"

    # Extract (req_key, pol) pairs from a rule
    def reqs_for_rule(rule: dict):
        out = []
        for req in rule.get("requirements", []) or []:
            k = pred_key(req)
            if not k:
                continue
            out.append((k, req_polarity(req), req))
        return out

    # Precompute apply formulas and per-req polarity for all existing rules
    existing = []
    for r in existing_rules:
        r_apply = apply_formula(r)
        for k, pol, raw in reqs_for_rule(r):
            existing.append((r, r_apply, k, pol, raw))

    # New rule payload
    new_apply = apply_formula(new_rule)
    new_reqs = reqs_for_rule(new_rule)

    issues = []

    # For each predicate in the new rule, compare against each existing rule's same predicate
    for (k_new, pol_new, raw_new) in new_reqs:
        for (r_old, apply_old, k_old, pol_old, raw_old) in existing:
            if k_new != k_old:
                continue  # only same predicate keys can contradict

            # If polarity is the same, skip (no contradiction by definition)
            if pol_new == pol_old:
                continue

            # Build a fresh solver per pair to ask: can both rules apply in the same world?
            s = Solver()
            s.add(*base_world_constraints)

            state_k = get_state_for(k_new)

            # Encode both implications for this predicate (single shared state)
            if pol_new == "require":
                s.add(Implies(new_apply, state_k == Require))
            else:
                s.add(Implies(new_apply, state_k == Forbid))

            if pol_old == "require":
                s.add(Implies(apply_old, state_k == Require))
            else:
                s.add(Implies(apply_old, state_k == Forbid))

            # Force both rules to apply simultaneously in this query world
            s.add(new_apply)
            s.add(apply_old)

            # If SAT, then a world exists where both apply AND the single state must be equal to both Require and Forbid.
            # That makes it UNSAT. So:
            # - UNSAT => true contradiction (cannot satisfy both) => report conflict.
            # - SAT   => scopes don't actually overlap in the same world => no conflict.
            res = s.check()
            if res != sat:
                # Strong conflict
                issues.append(
                    f"SMT conflict on predicate '{k_new}': new rule ('{raw_new}') "
                    f"({pol_new}) contradicts existing rule ('{raw_old}') ({pol_old}) under overlapping scopes."
                )

    return issues

def llm_judge_rule(client, model: str, rule: dict, source_text: str) -> list[str]:
    sys = {"role":"system",
           "content": """
            You are a strict compliance rule checker.

            GOAL
            Given (A) a JSON "rule" produced by an extractor and (B) its exact "source_text" clause, return a JSON object:
            {"issues":["<label1>", "<label2>", ...]}
            where each label is chosen from this CLOSED SET (must match exactly):
            - "Prohibition without concrete conditions"
            - "Missing hazard"
            - "Missing requirements"
            - "Scope fields empty"
            - "Conflicts between requirements and exceptions"

            Report ONLY problems that make operational testing unclear or invalid. If none apply, return {"issues":[]}.

            INPUTS
            - rule: JSON with fields: source, scope{actor,data_domain,context}, hazard, conditions, exceptions, requirements, evidence, severity
            - source_text: the original clause text for that rule

            GENERAL PRINCIPLES
            - Source-of-truth is source_text. Do NOT infer new facts beyond what a competent auditor could verify from the clause.
            - Prefer under-flagging to over-flagging. If a field is optional by schema and is correctly empty, do not flag it.

            DEFINITIONS & DECISION RULES (what each label means)

            1) "Prohibition without concrete conditions"
            Use when the rule encodes a ban/prohibition (e.g., "must not", "shall not", "may not") but conditions[] is empty OR lacks at least one specific trigger that makes the ban testable (e.g., "before deployment", "used for clinical decision support", "processing PHI").
            - DO flag if requirements[] contains only negative phrasing without a trigger in conditions[].
            - DO NOT flag if the rule is an affirmative obligation (not a prohibition).

            2) "Missing hazard"
            Use ONLY when the rule.hazard field is empty or all-whitespace after trimming.
            - Flag examples: "", "   ".
            - Do NOT consider the specificity/quality of the hazard for now.

            3) "Missing requirements"
            Use ONLY when rule.requirements is an empty array.
            - Flag example: [].
            - Do NOT evaluate actionability, testability, wording quality, or actor mapping for now.

            Examples
            Accept: "perform a bias review before deployment"; "disclose LMM use to end users"; "document known failure modes."
            Flag: "ensure fairness"; "be transparent"; "LMMs should be safe."


            4) "Scope fields empty"
            Use ONLY when scope.actor is empty OR all scope subfields are empty WHERE the clause clearly assigns a duty to an organizational actor (developer, provider, deployer, healthcare_org, clinician, user, data_controller, data_processor) but the rule failed to capture it.
            - Do NOT flag if the clause truly does not impose any org-facing duty (e.g., it speaks only to regulators) or if scope is correctly narrowed (e.g., actor set but data_domain/context legitimately empty).

            5) "Conflicts between requirements and exceptions"
            Use when exceptions[] negate or materially contradict the requirements so that the rule cannot be operationalized (e.g., requirement says "must do X" and exception says "unless not doing X is necessary" with no testable criteria).
            - Also use when multiple requirements contradict each other within the same rule.

            EDGE GUIDANCE
            - Evidence fields are irrelevant to this judge unless they create one of the above issues (e.g., they don’t).
            - Severity choice is not judged here unless it causes a listed issue (it doesn’t).

            OUTPUT FORMAT (strict)
            Return exactly one JSON object with the single key "issues". No prose, no Markdown, no extra keys.

            FEW-SHOT EXAMPLES

            Example 1 — Clean rule → no issues
            <INPUT>
            rule = {
            "source":{"doc":"D1","citation":"§3.2","span_id":"s10"},
            "scope":{"actor":["developer"],"data_domain":["health"],"context":["clinical"]},
            "hazard":"clinically unsafe hallucination",
            "conditions":["used for clinical decision support"],
            "exceptions":[],
            "requirements":["developer must validate model outputs against a clinical reference set before release"],
            "evidence":[],
            "severity":"high"
            }
            source_text = "When AI is used for clinical decision support, developers must validate outputs against clinical references to avoid unsafe hallucinations."
            </INPUT>
            <OUTPUT>
            {"issues":[]}
            </OUTPUT>

            Example 2 — Prohibition without concrete conditions
            <INPUT>
            rule = {
            "source":{"doc":"D1","citation":"§4.1","span_id":"s22"},
            "scope":{"actor":["provider"],"data_domain":[],"context":[]},
            "hazard":"unauthorized PHI disclosure",
            "conditions":[],
            "exceptions":[],
            "requirements":["provider must not release data externally"],
            "evidence":[],
            "severity":"high"
            }
            source_text = "Providers shall not release PHI externally without proper authorization."
            </INPUT>
            <OUTPUT>
            {"issues":["Prohibition without concrete conditions"]}
            </OUTPUT>

            Example 4 — Scope fields empty (actor missing despite org duty)
            <INPUT>
            rule = {
            "source":{"doc":"D1","citation":"§6.2","span_id":"s40"},
            "scope":{"actor":[],"data_domain":[],"context":[]},
            "hazard":"bias against protected groups",
            "conditions":[],
            "exceptions":[],
            "requirements":["must publish subgroup performance metrics"],
            "evidence":["impact assessment (subgroup metrics)"],
            "severity":"high"
            }
            source_text = "Deployers must publish subgroup performance metrics."
            </INPUT>
            <OUTPUT>
            {"issues":["Scope fields empty","Missing requirements"]}
            </OUTPUT>

            Example 5 — Conflicting requirement and exception
            <INPUT>
            rule = {
            "source":{"doc":"D1","citation":"§7.1","span_id":"s55"},
            "scope":{"actor":["deployer"],"data_domain":[],"context":[]},
            "hazard":"patient safety risk",
            "conditions":["before deployment"],
            "exceptions":["unless no validation is feasible"],
            "requirements":["deployer must validate the system before deployment"],
            "evidence":[],
            "severity":"high"
            }
            source_text = "Deployers must validate the system before deployment; an exception applies only with regulator approval when validation is infeasible."
            </INPUT>
            <OUTPUT>
            {"issues":["Conflicts between requirements and exceptions","Missing requirements"]}
            </OUTPUT>
            """    

    }
    user = {"role":"user","content": json.dumps({
        "rule": rule, "source_text": source_text,
        "checks": [
            "Prohibition without concrete conditions[]",
            "Missing hazard",
            "Missing requirements[]",
            "Missing Scope fields",
            "Conflicts between requirements[] and exceptions[]",
        ]}, ensure_ascii=False)
    }

    def _run(payload):
        # Prefer max_completion_tokens; retry with max_tokens if needed.
        mt = payload.pop("max_completion_tokens", None) or 10000
        payload["max_completion_tokens"] = mt
        payload.pop("temperature", None)
        try:
            r = client.chat.completions.create(**payload)
            return r.choices[0].message.content or ""
        except Exception as e:
            msg = getattr(e, "message", "") or str(e)
            if "max_completion_tokens" in msg or "Unsupported parameter" in msg:
                payload.pop("max_completion_tokens", None)
                payload["max_tokens"] = mt
                payload.pop("temperature", None)
                r = client.chat.completions.create(**payload)
                return r.choices[0].message.content or ""
            raise

    # A) json_object
    try:
        raw = _run({
            "model": model,
            "messages": [sys, user],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 10000,
        })
    except Exception:
        # B) plain JSON fallback
        raw = _run({
            "model": model,
            "messages": [sys, user],
            "max_completion_tokens": 10000,
        })

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("issues"), list):
            return [str(x).strip() for x in obj["issues"] if str(x).strip()]
    except Exception:
        pass
    return []



def generate_counterfactuals(client, model: str, text: str, k: int=0) -> List[str]:
    """Generate up to k tiny paraphrases (negation/quantifier flip)."""
    if k <= 0:
        return []
    sys = {"role":"system","content":"Produce minimal counterfactual paraphrases (negation/quantifier flip). Return a JSON array of strings."}
    user = {"role":"user","content": text}
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[sys, user],
            response_format={"type":"json_object"},
            max_completion_tokens=10000,
            # temperature=0
        )
        obj = json.loads(r.choices[0].message.content or "{}")
        if isinstance(obj, dict) and "paraphrases" in obj and isinstance(obj["paraphrases"], list):
            return obj["paraphrases"][:k]
        if isinstance(obj, list):
            return obj[:k]
    except Exception as e:
        _print_openai_error(e)
    return []

def polarity_signature(rules: List[dict]) -> Dict[str, bool]:
    neg = False
    for r in rules:
        txt = " ".join(r.get("requirements", [])).lower()
        if any(kw in txt for kw in ("must not", "may not", "shall not", "prohibit", "forbid", "forbidden")):
            neg = True
            break
    return {"neg": neg}


# ------------------------ Runner ------------------------
def run_hardened(candidates_jsonl: str,
                 out_schema_path: str,
                 out_rules_jsonl: Optional[str] = None,
                 provider: str = "openai",
                 model: Optional[str] = None,
                 # New optional knobs (all default off/compat-safe):
                 decoder: str = "jsonschema",
                 evidence_gate: bool = True,
                 allow_domains: Optional[List[str]] = None,
                 smt: bool = False,
                 repair_passes: int = 1,
                 counterfactuals: int = 0,
                 llm_judge: bool=False,
                 llm_judge_model: Optional[str]=None,
                 decode_retries: int = 3) -> None:

    t_run0 = time.perf_counter()
    logger.info("[Step3] init | provider=%s model=%s decoder=%s evidence_gate=%s allow_domains=%s smt=%s repair_passes=%d counterfactuals=%d llm_judge=%s decode_retries=%d",
                provider, model or "(auto)", decoder, evidence_gate, allow_domains, smt, repair_passes, counterfactuals, llm_judge, decode_retries)

    # Client + model
    client = _get_client_for_provider(provider)
    model_to_use = select_working_model(client, provider, user_model=model)
    logger.info("[Step3] using provider=%s model=%s", provider, model_to_use)

    out_rules_path = pathlib.Path(out_rules_jsonl or (pathlib.Path(out_schema_path).with_suffix(".jsonl")))
    out_rules_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[Step3] output: rules_jsonl=%s schema_out=%s", out_rules_path, out_schema_path)

    rules_accum: List[Dict[str, Any]] = []
    rules_accum_clean: List[Dict[str, Any]] = []   # <— add this
    seen_payloads: set[str] = set()
    spans_total = 0
    spans_success = 0
    spans_failed = 0
    rules_total_before_fix = 0
    rules_total_after_fix = 0

    with open(candidates_jsonl, "r", encoding="utf-8") as fin, \
         open(out_rules_path, "w", encoding="utf-8") as fout:

        for line in fin:
            t_span0 = time.perf_counter()
            cand = json.loads(line)
            text = (cand.get("text") or "").strip()
            if not text:
                continue

            span_id = cand.get("span_id", "")
            doc0 = (cand.get("doc") or "").strip()
            cit0 = (cand.get("citation") or "").strip()
            doc, citation = _fallback_doc_and_cit(cand, candidates_jsonl_path=candidates_jsonl)
            corpus = guess_corpus_name(doc or doc0)
            spans_total += 1

            logger.info("[Step3][%s] start | corpus=%s doc=%s citation=%s", span_id, corpus, (doc or doc0), (citation or cit0))
            window_text = text
            msgs = build_user_msg(window_text, doc or doc0, citation or cit0, span_id, corpus)

            try:
                # ----- Decode with retries -----
                last_err = None
                success = False
                arr = []
                for attempt in range(max(1, decode_retries)):
                    logger.info("[Step3][%s] decode attempt %d/%d", span_id, attempt+1, decode_retries)
                    t_dec0 = time.perf_counter()
                    try:
                        arr = decode_rules(client, model_to_use, msgs, decoder=decoder)
                        logger.debug("[Step3][%s] raw_arr_len=%d", span_id, len(arr))
                        arr = explode_composite_rules(arr)
                        rules_total_before_fix += len(arr)
                        rules = normalize_and_validate_rules(
                            arr,
                            {"doc": doc, "citation": citation, "span_id": span_id},
                            RULES_VALIDATOR
                        )
                        dt = time.perf_counter() - t_dec0
                        logger.info("[Step3][%s] decode OK in %.2fs | rules=%d", span_id, dt, len(rules))
                        success = True
                        break
                    except Exception as e:
                        dt = time.perf_counter() - t_dec0
                        last_err = e
                        logger.warning("[Step3][%s] decode attempt %d failed in %.2fs: %s", span_id, attempt+1, dt, e)
                        # tighten prompt once after first failure
                        if attempt == 0:
                            msgs = build_user_msg(
                                text + "\n\nSTRICT: If uncertain, return {\"rules\": []}. "
                                       "MUST include non-empty 'hazard'; fill scope.actor/data_domain/context "
                                       "using ONLY canonical tokens; include evidence[]. No prose.",
                                doc or doc0, citation or cit0, span_id, corpus
                            )
                        time.sleep(0.2)

                if not success:
                    spans_failed += 1
                    logger.error("[Step3][%s] giving up after %d attempts: %s", span_id, decode_retries, last_err)
                    fout.write(json.dumps({
                        "from_span": span_id, "doc": doc, "citation": citation,
                        "error": f"{type(last_err).__name__}: {last_err}"
                    }, ensure_ascii=False) + "\n")
                    continue

                # ----- Judge / Repair loop -----
                fixed_rules: List[tuple[dict, List[str]]] = []
                rule_idx = 0
                for r in rules:
                    rule_idx += 1
                    logger.debug("[Step3][%s][r#%d] initial scope=%s hazard=%s severity=%s reqs=%s",
                                 span_id, rule_idx, r.get("scope"), r.get("hazard"), r.get("severity"), r.get("requirements"))

                    issues = judge_rule(r, allow_domains=allow_domains, use_evidence_gate=evidence_gate)
                    if issues:
                        logger.info("[Step3][%s][r#%d] deterministic issues=%s", span_id, rule_idx, issues)

                    if llm_judge:
                        jmodel = llm_judge_model or model_to_use
                        t_j0 = time.perf_counter()
                        llm_issues = llm_judge_rule(client, jmodel, r, text)
                        dt_j = time.perf_counter() - t_j0
                        if llm_issues:
                            logger.info("[Step3][%s][r#%d] llm_judge issues=%s (%.2fs)", span_id, rule_idx, llm_issues, dt_j)
                        # merge + dedupe
                        _seen = set(); merged = []
                        for it in issues + llm_issues:
                            if it not in _seen:
                                _seen.add(it); merged.append(it)
                        issues = merged

                    passes = 0
                    while issues and passes < max(0, repair_passes):
                        logger.info("[Step3][%s][r#%d] repair pass %d/%d | issues_before=%s",
                                    span_id, rule_idx, passes+1, repair_passes, issues)
                        t_rep0 = time.perf_counter()
                        r = repair_rule_llm(client, model_to_use, r, issues, text)
                        r = normalize_and_validate_rules([r], {"doc":doc,"citation":citation,"span_id":span_id}, RULES_VALIDATOR)[0]

                        issues = judge_rule(r, allow_domains=allow_domains, use_evidence_gate=evidence_gate)
                        if llm_judge:
                            jmodel = llm_judge_model or model_to_use
                            llm_issues = llm_judge_rule(client, jmodel, r, text)
                            _seen = set(); merged = []
                            for it in issues + llm_issues:
                                if it not in _seen:
                                    _seen.add(it); merged.append(it)
                            issues = merged

                        dt_rep = time.perf_counter() - t_rep0
                        logger.info("[Step3][%s][r#%d] repair pass %d done in %.2fs | issues_after=%s",
                                    span_id, rule_idx, passes+1, dt_rep, issues)
                        passes += 1

                    if smt:
                        t_smt0 = time.perf_counter()
                        smt_issues = smt_check_conflicts(r, rules_accum)
                        dt_smt = time.perf_counter() - t_smt0
                        if smt_issues:
                            logger.warning("[Step3][%s][r#%d] SMT conflicts=%s (%.2fs)", span_id, rule_idx, smt_issues, dt_smt)
                            issues += smt_issues

                    fixed_rules.append((r, issues))

                # ----- Counterfactuals (optional) -----
                if counterfactuals > 0:
                    t_cf0 = time.perf_counter()
                    cfs = generate_counterfactuals(client, model_to_use, text, k=counterfactuals)
                    logger.info("[Step3][%s] counterfactuals generated=%d", span_id, len(cfs))
                    for i, cf in enumerate(cfs, 1):
                        logger.debug("[Step3][%s] CF#%d: %s", span_id, i, cf)
                        cf_msgs = build_user_msg(cf, doc or doc0, citation or cit0, span_id, corpus)
                        cf_arr = decode_rules(client, model_to_use, cf_msgs, decoder=decoder)
                        cf_arr = explode_composite_rules(cf_arr)
                        cf_rules = normalize_and_validate_rules(cf_arr, {"doc":doc,"citation":citation,"span_id":span_id}, RULES_VALIDATOR)
                        base_neg = polarity_signature([x for x,_ in fixed_rules]).get("neg", False)
                        cf_neg = polarity_signature(cf_rules).get("neg", False)
                        if base_neg == cf_neg and fixed_rules:
                            fixed_rules[0][1].append("Counterfactual sensitivity: no polarity change detected")
                            logger.info("[Step3][%s] CF#%d no polarity change → noted on first rule", span_id, i)
                    logger.info("[Step3][%s] counterfactuals done in %.2fs", span_id, time.perf_counter() - t_cf0)

                # ----- Assign IDs, dedupe, write -----
                written_rules = 0
                for _, (r, issues) in enumerate(fixed_rules, 1):
                    logging.info(r)
                    if not r.get("rule_id"):
                        prefix = _stable_id_prefix(doc, span_id)
                        r["rule_id"] = f"{prefix}-{_rule_hash(r)}"
                    key = json.dumps(r, sort_keys=True, ensure_ascii=False)
                    if key in seen_payloads:
                        logger.debug("[Step3][%s] skipping duplicate rule_id=%s", span_id, r.get("rule_id"))
                        continue
                    seen_payloads.add(key)
                    rules_accum.append(r)
                    if not issues:
                        rules_accum_clean.append(r) 
                    rules_total_after_fix += 1
                    fout.write(json.dumps({
                        "from_span": span_id, "doc": doc, "citation": citation,
                        "rule": r, "confidence": 0.75,
                        "verify_issues": issues
                    }, ensure_ascii=False) + "\n")
                    written_rules += 1

                spans_success += 1
                dt_span = time.perf_counter() - t_span0
                logger.info("[Step3][%s] done in %.2fs | wrote_rules=%d", span_id, dt_span, written_rules)

            except Exception as e:
                spans_failed += 1
                logger.warning("[Step3][%s] Extraction failed: %s", span_id, e)
                fout.write(json.dumps({
                    "from_span": span_id, "doc": doc, "citation": citation,
                    "error": f"{type(e).__name__}: {e}"
                }, ensure_ascii=False) + "\n")
                continue

            time.sleep(0.12)  # polite pacing

    # ----- Write schema file (YAML if possible; else JSON) -----
    t_schema0 = time.perf_counter()
    out_schema_path = str(out_rules_path.parent / pathlib.Path(out_schema_path).name)
    try:
        if HAVE_YAML and out_schema_path.lower().endswith((".yml",".yaml")):
            with open(out_schema_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(rules_accum_clean, f, sort_keys=False, allow_unicode=True)
            wrote = out_schema_path
            logger.info("[Step3] schema YAML written: %s", wrote)
        else:
            out_json = str(pathlib.Path(out_schema_path).with_suffix(".json"))
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(rules_accum_clean, f, ensure_ascii=False, indent=2)
            wrote = out_json
            logger.info("[Step3] schema JSON written: %s", wrote)
    except Exception as e:
        out_json = str(pathlib.Path(out_schema_path).with_suffix(".json"))
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rules_accum_clean, f, ensure_ascii=False, indent=2)
        wrote = out_json
        logger.warning("[Step3] schema write fallback to JSON due to %s → %s", e, wrote)

    dt_schema = time.perf_counter() - t_schema0
    dt_total = time.perf_counter() - t_run0
    logger.info("✓ Step 3 summary | rules_before_fix=%d rules_written=%d spans_total=%d ok=%d failed=%d schema_time=%.2fs total=%.2fs",
                rules_total_before_fix, rules_total_after_fix, spans_total, spans_success, spans_failed, dt_schema, dt_total)
    print(f"✓ Extracted {len(rules_accum_clean)} rules (errors={spans_failed})")
    print(f"→ Schema: {wrote}")
    print(f"→ Per-candidate: {out_rules_path}")

# =========================
# Example invocation (edit paths/model as needed)
# =========================
# import os
# os.environ["OPENAI_API_KEY"] = "sk-..."  # if not already set in your environment
# home = os.path.expanduser("~")
# candidates_path = os.path.join(home, "Downloads", "NIST.AI.100-1.candidates.jsonl")
# out_dir = pathlib.Path("./out"); out_dir.mkdir(parents=True, exist_ok=True)
# run_hardened(
#     candidates_jsonl=candidates_path,
#     out_schema_path=str(out_dir / "policy_schema.yaml"),
#     out_rules_jsonl=str(out_dir / "NIST.extracted.jsonl"),
#     model="gpt-5-mini",   # or None to auto-pick
#     decoder="grammar",     # "jsonschema" (default) or "grammar"
#     evidence_gate=True,    # require span_id + evidence[]
#     allow_domains=[".gov", "europa.eu"],  # optional
#     smt=True,              # enable simple contradiction check if z3 installed
#     repair_passes=1,       # minimal local repair attempts
#     counterfactuals=1      # generate 1 counterfactual probe
# )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_jsonl", required=True)
    ap.add_argument("--out_schema_path", required=True)
    ap.add_argument("--out_rules_jsonl")
    ap.add_argument("--provider", default="openai", choices=list(PROVIDERS.keys()))
    ap.add_argument("--model")
    ap.add_argument("--decoder", default="jsonschema", choices=["jsonschema","grammar"])
    ap.add_argument("--no_evidence_gate", action="store_true")
    ap.add_argument("--allow_domains", nargs="*")
    ap.add_argument("--smt", action="store_true")
    ap.add_argument("--repair_passes", type=int, default=1)
    ap.add_argument("--counterfactuals", type=int, default=0)
    ap.add_argument("--decode-retries", type=int, default=3,
                help="Max decode/validation retries per span (repair-on-parse).")
    args = ap.parse_args()

    run_hardened(
        candidates_jsonl=args.candidates_jsonl,
        out_schema_path=args.out_schema_path,
        out_rules_jsonl=args.out_rules_jsonl,
        provider=args.provider,
        model=args.model,
        decoder=args.decoder,
        evidence_gate=not args.no_evidence_gate,
        allow_domains=args.allow_domains,
        smt=args.smt,
        repair_passes=args.repair_passes,
        counterfactuals=args.counterfactuals,
        decode_retries=args.decode_retries,
    )
