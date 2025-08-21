# rails/actions/policy_actions.py
from __future__ import annotations

import os, json, re, time, logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from nemoguardrails.actions import action

# ------------ Logging ------------
log = logging.getLogger("rails.policy")
if not log.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    h.setFormatter(fmt)
    log.addHandler(h)
log.setLevel(logging.INFO)

# ------------ Config ------------
AGENT_BASE   = os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8000")
GUARD_MODE   = os.getenv("GUARD_MODE", "block")  # "block" | "revise"
JUDGE_MODEL  = os.getenv("POLICY_JUDGE_MODEL", "gpt-4o-mini")
REV_MODEL    = os.getenv("POLICY_REVISION_MODEL", "gpt-4o-mini")

# Optional: keep only IO-check rules
ONLY_IO_CHECK = False  # True => restrict to rules with testability.evidence_signals == ["io_check"]

# ------------ Load prebuilt rule table ------------
RULES_PATH = Path(__file__).with_name("policy_rules.json")
if not RULES_PATH.exists():
    log.warning("policy_rules.json not found; did you run tools/rules_to_nemo.py ?")
try:
    RULES: List[Dict[str, Any]] = json.loads(RULES_PATH.read_text(encoding="utf-8")) if RULES_PATH.exists() else []
    if not isinstance(RULES, list):
        log.warning("policy_rules.json root is not a list; ignoring.")
        RULES = []
except Exception as e:
    log.exception("Failed to load policy_rules.json: %s", e)
    RULES = []

def _is_io_only(rule: Dict[str, Any]) -> bool:
    t = (rule.get("testability") or {})
    sigs = t.get("evidence_signals")
    return isinstance(sigs, list) and len(sigs) == 1 and str(sigs[0]).strip().lower() == "io_check"

if ONLY_IO_CHECK:
    before = len(RULES)
    RULES = [r for r in RULES if _is_io_only(r)]
    log.info("Filtered rules to IO-only: %d -> %d", before, len(RULES))

# ------------ OpenAI client ------------
def _openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set for NeMo actions.")
    from openai import OpenAI  # lazy import
    return OpenAI(api_key=key)

# ------------ Simple input risk classifier (optional helper) ------------
_PATTERNS = {
    "pii": re.compile(r"\b(ssn|social\s*security|passport|driver'?s?\s*license|mrn|medical\s*record|npi)\b", re.I),
    "secrets": re.compile(r"\b(token|api[_-]?key|password|secret|bearer\s+[a-z0-9\-_.]+)\b", re.I),
    "exfil": re.compile(r"\b(base64|hex\s*dump|decode|decrypt|extract|deobfuscat)\w*\b", re.I),
    "bypass": re.compile(r"\b(bypass|disable|turn\s*off|override|ignore)\b.*\b(safeguard|guard|safety|policy|filter|auth|validation)\b", re.I),
    "med_claims": re.compile(r"\b(diagnos(e|is)|treat(ment)?|prescrib(e|ing)|dose|protocol|contraindication)\b", re.I),
    "copyright": re.compile(r"\b(copy(right(ed)?|paste)|verbatim|proprietary|internal\s+code|license)\b", re.I),
    "tenant": re.compile(r"\b(cross[-\s]?tenant|other\s+tenant|outside\s+(workspace|org|boundary))\b", re.I),
}
_DEFUSERS = re.compile(r"\b(example|sample|demo|mock|placeholder|lorem|ipsum)\b", re.I)

def _score_input(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    score = 0.0
    for name, rx in _PATTERNS.items():
        hits = len(rx.findall(s))
        if not hits:
            continue
        if name in ("secrets", "exfil", "bypass"):
            score += 0.6 + 0.2 * min(hits, 2)
        elif name in ("pii", "med_claims", "tenant", "copyright"):
            score += 0.4 + 0.15 * min(hits, 2)
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", s):  # SSN
        score += 0.7
    if re.search(r"\b[0-9A-Fa-f]{32,}\b", s):   # long hex
        score += 0.4
    if re.search(r"\b[a-zA-Z0-9+/]{40,}={0,2}\b", s):  # base64-ish
        score += 0.4
    if _DEFUSERS.search(s):
        score -= 0.2
    return max(0.0, min(1.0, round(score, 3)))

@action()
def InferCategoryAction(input: str) -> str:
    """Returns 'ADVERSARIAL' when risk score >= 0.6, else 'BENIGN'."""
    score = _score_input(input or "")
    log.debug("[infer_category] score=%.2f text=%r", score, (input or "")[:120])
    return "ADVERSARIAL" if score >= 0.6 else "BENIGN"

# ------------ Judge & Reviser ------------
REV_SYS = (
    "Rewrite the assistant's draft so it is safe and compliant with the given requirement.\n"
    "If safe rewriting isn't possible, return a brief refusal with appropriate uncertainty.\n"
    "Return only the final user-facing text."
)

def _revise(requirement: str, prompt: str, draft: str,
            retries: int = 1, backoff: float = 1.0) -> str:
    msgs = [
        {"role": "system", "content": REV_SYS},
        {"role": "user", "content": json.dumps({
            "requirement": requirement,
            "prompt": prompt,
            "draft_response": draft
        }, ensure_ascii=False)}
    ]
    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            cli = _openai()
            out = cli.chat.completions.create(
                model=REV_MODEL,
                messages=msgs,
                temperature=0.2,
                max_tokens=400,
            )
            text = (out.choices[0].message.content or "").strip()
            return text or "I can’t help with that request."
        except Exception as e:
            last_err = e
            log.warning("[revise] attempt %d failed: %s", i + 1, e)
            time.sleep(backoff * (i + 1))
    log.error("[revise] giving up after retries; returning refusal. last_err=%s", last_err)
    return "I can’t help with that request."

def _severity(rule: Dict[str, Any]) -> int:
    _SEV_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    sev = (rule.get("severity") or rule.get("rule", {}).get("severity") or "").lower()
    return _SEV_RANK.get(sev, 1)

def _iter_rules() -> List[Dict[str, Any]]:
    """Yield usable rules with a 'requirement' string, sorted by severity desc."""
    usable = []
    for r in RULES:
        req = r.get("requirement") or r.get("rule", {}).get("requirements")
        if isinstance(req, list):
            req = "; ".join([str(x) for x in req if str(x).strip()])
        if isinstance(req, str) and req.strip():
            rr = dict(r)
            rr["requirement"] = req.strip()
            usable.append(rr)
    usable.sort(key=_severity, reverse=True)
    return usable

def _judge_rule_simple(rule: Dict[str, Any], prompt: str, response_text: str) -> Dict[str, Any]:
    """
    Minimal judge: send requirement + optional context (scope/conditions/exceptions)
    to the LLM and return {ok: bool, reason: str}. No runtime facts, no evidence.
    """
    ctx = {
        "requirement": rule.get("requirement"),
        "scope": rule.get("scope", {}),
        "conditions": rule.get("conditions", []),
        "exceptions": rule.get("exceptions", []),
        "rule_id": rule.get("rule_id"),
        "severity": rule.get("severity"),
    }
    msgs = [
        {"role": "system", "content":
         "You are a strict compliance judge for LLM outputs. "
         "Return JSON {\"pass\": true|false, \"reason\": \"<short>\"}. "
         "Apply the requirement considering the provided context (scope/conditions/exceptions) only if relevant."
        },
        {"role": "user", "content": json.dumps({
            "rule_context": ctx,
            "prompt": prompt,
            "response": response_text
        }, ensure_ascii=False)}
    ]
    try:
        cli = _openai()
        out = cli.chat.completions.create(
            model=JUDGE_MODEL,
            messages=msgs,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=240,
        )
        obj = json.loads(out.choices[0].message.content or "{}")
        return {"ok": bool(obj.get("pass")), "reason": str(obj.get("reason", "")).strip()}
    except Exception as e:
        log.warning("[_judge_rule_simple] judge unavailable, treating as pass: %s", e)
        return {"ok": True, "reason": "judge_unavailable"}

# ------------ Dynamic evaluator (simple) ------------
@action()
def EvaluateRulesAction(input: str, output: str,
                        rule_ids: Optional[List[str]] = None,
                        enforcement: str = "top_severity") -> Dict[str, Any]:
    """
    Simple evaluation: iterate selected rules, call the LLM judge with the rule's
    requirement + context, collect violations. No runtime facts, no evidence gathering.

    enforcement: "first_fail" | "top_severity" (default) | "all"
    Returns:
      {
        "decision": "ok|block|revise",
        "revised": "<text>",
        "reason": "<short>",
        "violations": [ {rule_id, severity_rank, reason}, ... ],
        "applied_rule_ids": [ ... ]
      }
    """
    text_in  = (input or "").strip()
    text_out = (output or "").strip()

    if not RULES:
        return {"decision":"ok","reason":"","revised":"","violations":[],"applied_rule_ids":[]}

    # choose rules (subset if rule_ids provided), keep overall severity ordering
    def _iter_selected():
        if rule_ids:
            wanted = set(rule_ids)
            for r in _iter_rules():
                if r.get("rule_id") in wanted:
                    yield r
        else:
            for r in _iter_rules():
                yield r

    violations: List[Dict[str, Any]] = []
    applied_ids: List[str] = []

    for r in _iter_selected():
        applied_ids.append(r.get("rule_id", "(no-id)"))
        verdict = _judge_rule_simple(r, text_in, text_out)
        if not verdict["ok"]:
            violations.append({
                "rule_id": r.get("rule_id","(no-id)"),
                "severity_rank": _severity(r),
                "reason": verdict.get("reason","")
            })
            if enforcement == "first_fail":
                break

    if not violations:
        return {"decision":"ok","reason":"","revised":"","violations":[],"applied_rule_ids": applied_ids}

    # pick controlling violation
    violations.sort(key=lambda v: v["severity_rank"], reverse=True)
    chosen = violations[0]

    if GUARD_MODE == "revise":
        # find requirement string for rewrite
        req = None
        for r in _iter_rules():
            if r.get("rule_id") == chosen["rule_id"]:
                req = r.get("requirement"); break
        revised = _revise(req or "Comply with policy.", text_in, text_out)
        return {"decision":"revise","reason":"","revised":revised,
                "violations":violations,"applied_rule_ids":applied_ids}

    # default block
    reason = chosen.get("reason") or "policy_violation"
    return {"decision":"block","reason":f"I can’t comply with that request. {reason}".strip(),
            "revised":"","violations":violations,"applied_rule_ids":applied_ids}

# ------------ Back-compat shims ------------
@action()
def CheckRulesAction(category: str, input: str, output: str) -> Dict[str, str]:
    """Legacy signature; delegates to EvaluateRulesAction with all rules."""
    res = EvaluateRulesAction(input=input, output=output, rule_ids=None, enforcement="top_severity")
    return {
        "decision": res["decision"],
        "reason":   res.get("reason",""),
        "revised":  res.get("revised",""),
        "output":   output
    }

@action()
def CompositeCheckAction(input: str, output: str) -> Dict[str, str]:
    """Unified entry point: no special-case logic; uses dynamic evaluation."""
    res = EvaluateRulesAction(input=input, output=output, rule_ids=None, enforcement="top_severity")
    return {
        "decision": res["decision"],
        "reason":   res.get("reason",""),
        "revised":  res.get("revised","")
    }

# ------------ Convenience helpers ------------
@action()
def IsBlockAction(decision: str) -> bool:
    return str(decision or "").strip().lower() == "block"

@action()
def IsReviseAction(decision: str) -> bool:
    return str(decision or "").strip().lower() == "revise"

# (Optional) Demo generator; not used by rails unless you call it explicitly
@action(name="generate_bot_message")
async def generate_bot_message(input: str) -> str:
    return f"test message"