# clause_miner.py
from __future__ import annotations
import os, re, json, hashlib
from collections import defaultdict
from typing import List

# -------- Patterns (expanded) --------
DEONTIC_CORE = [
    r"shall(?:\s+not)?", r"must(?:\s+not)?", r"should(?:\s+not)?",
    r"may\s+not", r"may\s+only\b",
    r"MUST|SHALL|SHOULD",
    r"is\s+required\s+to", r"are\s+required\s+to",
    r"is\s+prohibited\s+from", r"are\s+prohibited\s+from",
    r"forbidden", r"prohibited", r"unlawful",
    r"ensure\s+that", r"verify\s+that",
    r"enable\s+.*\s+only\s+if", r"guarantee\s+that",
    r"responsible\s+for", r"obliged\s+to", r"mandated\s+to"
]
DEONTIC_RE = re.compile(r"\b(" + r"|".join(DEONTIC_CORE) + r")\b", re.IGNORECASE)

EXCEPTION_CUES = [
    r"except", r"unless", r"provided\s+that", r"subject\s+to",
    r"notwithstanding", r"as\s+permitted\s+by", r"save\s+as",
    r"only\s+if", r"if\s", r"when\s", r"where\s"
]
EXCEPTION_RE = re.compile(r"\b(" + r"|".join(EXCEPTION_CUES) + r")\b", re.IGNORECASE)

EXEMPTION_RE = re.compile(r"\bshall\s+not\s+be\s+required\s+to\b", re.IGNORECASE)
DEFINITION_RE = re.compile(r"\b(means|is\s+defined\s+as|refers\s+to|denotes|includes|comprises|consists\s+of)\b", re.IGNORECASE)
DEADLINE_RE  = re.compile(r"\b(no\s+later\s+than|within)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b", re.IGNORECASE)
THRESHOLD_RE = re.compile(r"\b(more\s+than|at\s+least|no\s+more\s+than|not\s+exceed(?:ing)?)\s+([\d,]+(?:\.\d+)?)\s*(users?|requests?|percent|%)\b", re.IGNORECASE)
JURIS_RE = re.compile(r"\b(to\s+the\s+extent\s+permitted\s+by\s+law|unless\s+otherwise\s+required\s+by)\b", re.IGNORECASE)
CROSSREF_RE = re.compile(
    r"\b(Article\s+\d+(?:\(\d+\))?|Art\.\s*\d+(?:\(\d+\))?|Annex\s+[A-Z]|Appendix\s+[A-Z]|§\s?\d{3,}(?:\.\d+)*(?:\([a-z0-9]+\))*|Section\s+\d+(?:\.\d+)*)",
    re.IGNORECASE
)
BULLET_RE = re.compile(r"^\s*(?:[\-\u2022•\*]|\(?[a-zA-Z]\)|\(?[ivxIVX]+\)|\(?\d+\)|\d+\.)\s+")
INFORMATIVE_HINTS = re.compile(r"\b(example|examples|note|notes|guidance|informative|appendix|annex|foreword|acknowledg(e|ment))\b", re.IGNORECASE)
NORMATIVE_HINTS   = re.compile(r"\b(scope|requirements?|shall|must|obligations?)\b", re.IGNORECASE)
ADMIN_RE = re.compile(
    r"\b(this\s+(document|publication|playbook)|email\s+to|comments?\s+may\s+be\s+sent|"
    r"contact\s|printing\s+office|citation\s+information)\b",
    re.IGNORECASE
)
EPISTEMIC_MAYNOT_RE = re.compile(
    r"\bmay\s+not\s+(align|recognize|apply|capture|reflect|generalize|work|hold)\b",
    re.IGNORECASE
)
ACTOR_RE = re.compile(r"\b(organization(?:s)?|provider(?:s)?|deployer(?:s)?|user(?:s)?|operator(?:s)?|controller(?:s)?|processor(?:s)?|ai\s+actor(?:s)?)\b", re.IGNORECASE)

def _h8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8]

# -------- Heuristics --------
def _is_normative_section(section_path: List[str]) -> bool:
    if not section_path:
        return True
    joined = " / ".join([x or "" for x in section_path])
    if INFORMATIVE_HINTS.search(joined) and not NORMATIVE_HINTS.search(joined):
        return False
    return True

def _classify_type(text: str) -> str:
    t = text.lower()
    if EXEMPTION_RE.search(t): return "exemption"
    if EXCEPTION_RE.search(t): return "exception"
    if re.search(r"\b(shall|must)\s+not\b|prohibit|forbid|unlawful", t): return "prohibition"
    if DEFINITION_RE.search(t): return "definition"
    if (DEONTIC_RE.search(t) and not re.search(r"\bmay\s+(?!only|not\b)", t)):  # ignore plain "may"
        return "obligation"
    return "other"

def _severity(text: str) -> str:
    if re.search(r"\b(MUST|SHALL|PROHIBITED|UNLAWFUL)\b", text): return "high"
    if re.search(r"\b(shall|must|prohibit|unlawful|required)\b", text, re.I):   return "high"
    if re.search(r"\b(SHOULD|recommended)\b", text):                            return "medium"
    return "medium"

def _crossrefs(text: str) -> list:
    return list({m.group(0).strip() for m in CROSSREF_RE.finditer(text)})

def _scrape_meta(text: str) -> dict:
    meta = {}
    dl = DEADLINE_RE.search(text)
    if dl:
        meta["deadline"] = {"cue": dl.group(1), "value": int(dl.group(2)), "unit": dl.group(3)}
    th = THRESHOLD_RE.search(text)
    if th:
        meta["threshold"] = {"cmp": th.group(1), "value": th.group(2), "unit": th.group(3)}
    if JURIS_RE.search(text):
        meta["jurisdictional"] = True
    return meta

def _conf(text: str, ttype: str, cues: list, normative: bool) -> float:
    score = 0.4 if ttype in ("obligation","prohibition","exception","exemption") else 0.2
    score += min(0.2, 0.05 * len(cues))
    if normative: score += 0.2
    if ttype == "definition": score -= 0.2
    if "example:" in text.lower() or "“" in text or "”" in text: score -= 0.05
    return max(0.05, min(0.95, round(score, 2)))

def _ctx(rows, i, window=2):
    prevs, nexts = [], []
    for k in range(1, window+1):
        if i-k >= 0: prevs.append(rows[i-k].get("text",""))
        if i+k < len(rows): nexts.append(rows[i+k].get("text",""))
    prevs.reverse()
    return prevs, nexts

# -------- Utility: output path (same folder + suffix) --------
def default_out_path(inp_jsonl: str, suffix: str = "_candidates_filtered.jsonl") -> str:
    inp_abs = os.path.abspath(inp_jsonl)
    folder, base = os.path.split(inp_abs)
    stem, _ = os.path.splitext(base)
    return os.path.join(folder, f"{stem}{suffix}")

# -------- Main mining function (now returns saved path) --------
def mine(inp_jsonl: str, out_jsonl: str, *, keep_informative: bool=False,
         include_tables: bool=False, ctx_window: int=2,
         use_agent_mining: bool=False, agent_confidence_threshold: float=0.6) -> str:
    """
    Mine policy clause candidates from a spans JSONL file.

    Parameters
    ----------
    use_agent_mining : bool
        When True, spans that the regex heuristic classifies as "other"
        (would normally be dropped) but that still contain a deontic keyword
        are passed to PolicyMiningAgent for a second opinion.  This recovers
        implicit or paraphrased obligations that the regex misses.
        Requires OPENAI_API_KEY.  Default: False (zero API calls, original behaviour).
    agent_confidence_threshold : float
        Only consult the agent for spans whose heuristic confidence score is
        below this value.  Default 0.6 — the grey zone between a clear reject
        and a clear accept.
    """
    # Lazy-init the agent once per mine() call (avoids per-span overhead)
    _mining_agent = None
    if use_agent_mining:
        try:
            from agents.langchain_agents import PolicyMiningAgent
            _mining_agent = PolicyMiningAgent()
        except Exception as _e:
            import warnings
            warnings.warn(
                f"[step2] PolicyMiningAgent unavailable ({_e}); "
                "falling back to regex-only mining.",
                stacklevel=2,
            )

    buckets = defaultdict(list)
    with open(inp_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            r = json.loads(line)

            # Fallback doc_id from span_id if missing
            if not r.get("doc_id") and r.get("span_id") and isinstance(r["span_id"], str):
                r["doc_id"] = r["span_id"].split("/", 1)[0]

            kind = (r.get("kind") or "sentence").lower()
            if kind not in ("sentence","table"):
                continue
            if kind == "table" and not include_tables:
                continue
            key = (r.get("doc_id") or "", tuple(r.get("section_path") or []))
            buckets[key].append(r)

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    out = open(out_jsonl, "w", encoding="utf-8")
    seen = set()
    total = 0

    for (doc_id, sec_path), rows in buckets.items():
        rows.sort(key=lambda x: (x.get("page", 0), x.get("sent_index", 0)))
        normative_section = _is_normative_section(list(sec_path))
        last_deontic_idx = None
        last_deontic_type = None

        for i, r in enumerate(rows):
            text = (r.get("text") or "").strip()
            if not text:
                continue
            if ADMIN_RE.search(text) or EPISTEMIC_MAYNOT_RE.search(text):
                continue

            local_norm = normative_section
            if not keep_informative:
                if INFORMATIVE_HINTS.search(text) and not NORMATIVE_HINTS.search(text):
                    local_norm = False

            cues = []
            for pat in (DEONTIC_RE, EXCEPTION_RE, EXEMPTION_RE):
                if pat.search(text):
                    cues.append(pat.pattern[:28])

            ttype = _classify_type(text)

            inherited_from = None
            if ttype in ("other","definition") and BULLET_RE.match(text) and last_deontic_idx is not None:
                ttype = last_deontic_type or "obligation"
                inherited_from = rows[last_deontic_idx].get("span_id")

            if DEONTIC_RE.search(text) or EXCEPTION_RE.search(text) or EXEMPTION_RE.search(text):
                last_deontic_idx = i
                last_deontic_type = ttype

            keep = ttype in ("obligation","prohibition","exception","exemption") or (ttype=="definition" and local_norm)

            # ── PolicyMiningAgent: second-opinion for borderline spans ──────────
            # A span lands here as "other" when regex found deontic keywords but
            # _classify_type() couldn't commit to a canonical type.  The agent
            # deliberates with four tools (deontic check → section normative check
            # → clause type → actor presence) and may promote the span.
            if not keep and _mining_agent is not None and DEONTIC_RE.search(text):
                conf_score = _conf(text, ttype, cues, local_norm)
                if conf_score < agent_confidence_threshold:
                    prevs_tmp, nexts_tmp = _ctx(rows, i, window=ctx_window)
                    try:
                        decision = _mining_agent.run(
                            text=text,
                            section_path=list(sec_path),
                            span_id=r.get("span_id", ""),
                            context_prev=" ".join(prevs_tmp),
                            context_next=" ".join(nexts_tmp),
                        )
                        if decision.get("is_candidate"):
                            keep = True
                            ttype    = decision.get("clause_type", ttype)
                            severity = decision.get("severity",    severity)
                    except Exception as _agent_err:
                        pass  # agent failure is non-fatal; keep original decision
            # ──────────────────────────────────────────────────────────────────

            if not keep:
                continue

            if ttype == "prohibition" and not ACTOR_RE.search(text):
                continue

            prevs, nexts = _ctx(rows, i, window=ctx_window)

            severity = _severity(text)
            xrefs = _crossrefs(text)
            meta = _scrape_meta(text)
            meta["normative"] = bool(local_norm)

            keyhash = (_h8(text), ttype, severity, bool(local_norm))
            if keyhash in seen:
                continue
            seen.add(keyhash)

            rec = {
                "span_id": r.get("span_id"),
                "doc_id": doc_id,
                "section_path": list(sec_path),
                "page": r.get("page"),
                "text": text,
                "context_prev": prevs,
                "context_next": nexts,
                "type": ttype,
                "severity": severity,
                "cues": cues,
                "cross_refs": xrefs,
                "jurisdictional": bool(meta.get("jurisdictional", False)),
                "deadline": meta.get("deadline"),
                "threshold": meta.get("threshold"),
                "normative": meta["normative"],
                "inherited_from": inherited_from,
                "confidence": _conf(text, ttype, cues, local_norm)
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1

    out.close()
    # print/log if you want; function now returns the saved path
    # print(f"[miner-final] wrote {total} candidate clauses → {out_jsonl}")
    return out_jsonl

# -------- Convenience wrapper for imports --------
def run_clause_miner(inp_jsonl: str, *,
                     keep_informative: bool=False,
                     include_tables: bool=False,
                     ctx_window: int=2,
                     suffix: str="_candidates_filtered.jsonl",
                     use_agent_mining: bool=False,
                     agent_confidence_threshold: float=0.6) -> str:
    """
    Derives the output path next to `inp_jsonl` with the given suffix,
    runs mining, and returns the saved output path.

    Pass use_agent_mining=True to enable PolicyMiningAgent for borderline spans.
    """
    outp = default_out_path(inp_jsonl, suffix=suffix)
    return mine(inp_jsonl, outp, keep_informative=keep_informative,
                include_tables=include_tables, ctx_window=ctx_window,
                use_agent_mining=use_agent_mining,
                agent_confidence_threshold=agent_confidence_threshold)
