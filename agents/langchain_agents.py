"""
agents/langchain_agents.py
==========================
LangChain agent layer for the Policy-to-Tests (P2T) pipeline.

WHY AGENTS INSTEAD OF PLAIN LLM CALLS?
────────────────────────────────────────
The existing pipeline makes single-shot LLM calls (Steps 2, 3, 5). That
works for clear-cut cases but loses transparency and robustness on edge cases:
  - A borderline clause that looks like a definition but has an implicit obligation
  - A rule that is "actionable" but the oracle doesn't exist yet (not observable)
  - A JSON field that is a string when the schema requires an array

Agents solve this by breaking each decision into discrete tool calls, producing
an inspectable chain-of-thought, and retrying at the right level of granularity
(fix *only* the broken field, not re-generate the whole rule).

THREE AGENTS
────────────
┌──────────────────────┬──────────────────────────────────────────────────────┐
│ Agent                │ Plugs into                                           │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ PolicyMiningAgent    │ Step 2 — step2_clause_miner.py                       │
│                      │ Handles spans that regex heuristics mark as           │
│                      │ ambiguous (confidence < threshold).                   │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ TestabilityJudgeAgent│ Step 5 — step5_llm_as_judge_testability.py           │
│                      │ Replaces the single-shot _chat_json() call with a    │
│                      │ deliberate A→B→C multi-tool judgment.                 │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ SchemaRepairAgent    │ Step 3 — step3_llm_generation_few_shot.py            │
│                      │ Replaces ad-hoc string repairs with targeted,         │
│                      │ schema-validation-guided field fixes.                 │
└──────────────────────┴──────────────────────────────────────────────────────┘

INSTALL
───────
pip install langchain langchain-openai langchain-core

QUICK START
───────────
    import os
    os.environ["OPENAI_API_KEY"] = "sk-..."

    from agents.langchain_agents import (
        PolicyMiningAgent,
        TestabilityJudgeAgent,
        SchemaRepairAgent,
    )

    # ── Mining ──────────────────────────────────────────────────────────────
    miner = PolicyMiningAgent()
    decision = miner.run(
        text="The provider must retain audit logs for at least 90 days.",
        section_path=["Requirements", "Logging"],
        span_id="doc/p12/s3/span7",
        context_prev="",
        context_next="",
    )
    # decision = {"is_candidate": True, "clause_type": "obligation",
    #             "severity": "high", "confidence": 0.92, "reason": "..."}

    # ── Judge ────────────────────────────────────────────────────────────────
    judge = TestabilityJudgeAgent()
    verdict = judge.run(rule={
        "requirements": ["Providers must log every inference call with a correlation ID."],
        "hazard": "Missing audit trail for compliance review",
        "scope": {"actor": ["provider"], "data_domain": ["telemetry"], "context": ["prod"]},
    })
    # verdict = {"testable": True, "reason": "...", "evidence_signals": ["log_check"]}

    # ── Repair ───────────────────────────────────────────────────────────────
    repair = SchemaRepairAgent()
    fixed = repair.run(broken_rule={
        "rule_id": None,
        "source": {"doc": "HIPAA", "citation": "§164.312", "span_id": "hipaa/p3/s1"},
        "scope": {"actor": "provider",           # ← string, should be list
                  "data_domain": ["PHI"],
                  "context": []},
        "hazard": "Unauthorised PHI access",
        "conditions": [],
        "exceptions": [],
        "requirements": ["Encrypt PHI at rest"],
        "evidence": [],
        "severity": "CRITICAL",                  # ← wrong case
    })
    # fixed = {... valid rule dict conforming to RULE_SCHEMA ...}
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── make root importable so step2_clause_miner patterns can be reused ───────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── LangChain imports ────────────────────────────────────────────────────────
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError as _e:
    raise ImportError(
        "LangChain dependencies missing. Run:\n"
        "  pip install langchain langchain-openai langchain-core"
    ) from _e

# ── Re-use existing pipeline regex patterns ──────────────────────────────────
try:
    from step2_clause_miner import (
        DEONTIC_RE,
        EXCEPTION_RE,
        EXEMPTION_RE,
        DEFINITION_RE,
        DEADLINE_RE,
        THRESHOLD_RE,
        ACTOR_RE,
        INFORMATIVE_HINTS,
        NORMATIVE_HINTS,
        _classify_type,
        _severity,
        _crossrefs,
        _is_normative_section,
    )
    _MINER_AVAILABLE = True
except ImportError:
    _MINER_AVAILABLE = False

# ── Re-use schema from Step 3 ────────────────────────────────────────────────
try:
    from step3_llm_generation_few_shot import (
        RULE_SCHEMA,
        ACTOR_ENUM,
        DOMAIN_ENUM,
        CONTEXT_ENUM,
        CANON,
        FALLBACK,
        canonicalize_scope,
    )
    from jsonschema import Draft202012Validator
    _SCHEMA_AVAILABLE = True
    _RULE_VALIDATOR = Draft202012Validator(RULE_SCHEMA)
except ImportError:
    _SCHEMA_AVAILABLE = False
    _RULE_VALIDATOR = None

# ── Evidence signal vocabulary (mirrors Step 5) ──────────────────────────────
EVIDENCE_SIGNALS = {
    "io_check",
    "log_check",
    "config_check",
    "ci_gate",
    "data_check",
    "repo_check",
}

SEVERITY_LEVELS = {"low", "medium", "high", "critical"}


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Build a ChatOpenAI instance, raising early if no key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def _make_executor(llm: ChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
    """
    Build a LangChain tool-calling AgentExecutor.

    Uses create_tool_calling_agent (recommended over ReAct for GPT-4+ models)
    because tool-calling agents natively return structured JSON via function
    calling — no prompt-engineering for output format needed.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=8)


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT 1 — PolicyMiningAgent
#  WHERE:  Called from step2_clause_miner.py for low-confidence spans.
#  WHY:    Regex gives a binary signal. For ambiguous spans (confidence < 0.6),
#          a deliberate multi-tool agent can reason over deontic language,
#          section context, and actor presence before deciding to emit a
#          candidate — reducing false positives going into Step 3.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def check_deontic_language(text: str) -> str:
    """
    Scan the text for deontic keywords (shall, must, may not, prohibited, …).
    Returns a JSON object: {"found": bool, "matches": [list of matched tokens]}.
    Use this first to see whether there is any obligation or prohibition language.
    """
    if _MINER_AVAILABLE:
        matches = [m.group(0) for m in DEONTIC_RE.finditer(text)]
    else:
        _fallback = re.compile(
            r"\b(shall(?:\s+not)?|must(?:\s+not)?|may\s+not|prohibited|required|forbidden)\b",
            re.IGNORECASE,
        )
        matches = [m.group(0) for m in _fallback.finditer(text)]
    return json.dumps({"found": bool(matches), "matches": matches})


@tool
def classify_clause_type(text: str) -> str:
    """
    Classify the clause as one of: obligation / prohibition / exception /
    exemption / definition / other.
    Returns a JSON object: {"clause_type": str}.
    Call this after check_deontic_language confirms deontic language is present.
    """
    if _MINER_AVAILABLE:
        ct = _classify_type(text)
    else:
        t = text.lower()
        if re.search(r"\bshall\s+not\s+be\s+required\b", t):
            ct = "exemption"
        elif re.search(r"\b(except|unless|provided\s+that)\b", t):
            ct = "exception"
        elif re.search(r"\b(shall|must)\s+not\b|prohibit|forbid", t):
            ct = "prohibition"
        elif re.search(r"\b(means|is\s+defined\s+as|refers\s+to)\b", t):
            ct = "definition"
        elif re.search(r"\b(shall|must|required|obliged)\b", t, re.I):
            ct = "obligation"
        else:
            ct = "other"
    return json.dumps({"clause_type": ct})


@tool
def check_normative_section(section_path_json: str) -> str:
    """
    Given a JSON array of section path strings (e.g. '["Requirements","Logging"]'),
    determine whether the section is normative (should produce candidates) or
    informative (appendix, example, foreword — skip).
    Returns: {"is_normative": bool}.
    Call this when you are unsure whether the surrounding section context makes
    the span worth emitting.
    """
    try:
        section_path = json.loads(section_path_json)
    except Exception:
        section_path = []
    if _MINER_AVAILABLE:
        result = _is_normative_section(section_path)
    else:
        joined = " / ".join(section_path)
        informative = re.search(
            r"\b(example|note|guidance|informative|appendix|foreword)\b",
            joined, re.IGNORECASE
        )
        normative = re.search(r"\b(scope|requirements?|shall|must)\b", joined, re.IGNORECASE)
        result = not (informative and not normative)
    return json.dumps({"is_normative": result})


@tool
def extract_actor_hints(text: str) -> str:
    """
    Scan the text for actor mentions (organization, provider, deployer, user, …).
    Returns: {"actors_found": [list of raw matched strings]}.
    Use this to confirm the clause references a specific actor before emitting it.
    """
    if _MINER_AVAILABLE:
        actors = [m.group(0) for m in ACTOR_RE.finditer(text)]
    else:
        _a = re.compile(
            r"\b(organization|provider|deployer|user|operator|controller|processor)\b",
            re.IGNORECASE,
        )
        actors = [m.group(0) for m in _a.finditer(text)]
    return json.dumps({"actors_found": list(set(actors))})


@tool
def assess_severity(text: str) -> str:
    """
    Assess severity of the clause based on deontic strength.
    Returns: {"severity": "high"|"medium"|"low"}.
    Use this after classifying the clause type to determine its severity tier.
    """
    if _MINER_AVAILABLE:
        sev = _severity(text)
    else:
        if re.search(r"\b(MUST|SHALL|PROHIBITED|UNLAWFUL)\b", text):
            sev = "high"
        elif re.search(r"\b(shall|must|prohibit|required)\b", text, re.I):
            sev = "high"
        else:
            sev = "medium"
    return json.dumps({"severity": sev})


_MINING_TOOLS = [
    check_deontic_language,
    classify_clause_type,
    check_normative_section,
    extract_actor_hints,
    assess_severity,
]

_MINING_SYSTEM = """\
You are a policy clause classifier for an AI governance pipeline.

Your task: decide whether a given text span should be emitted as a policy
candidate clause for further LLM-based rule extraction (Step 3).

DECISION CRITERIA (all must be considered):
1. Is deontic language present? (use check_deontic_language)
2. Is the enclosing section normative, not informative? (use check_normative_section)
3. Is the clause type substantive (obligation / prohibition / exception)?
   Definitions and "other" types usually should NOT be emitted. (use classify_clause_type)
4. Is there at least one actor hint present? (use extract_actor_hints)
5. What severity does the language imply? (use assess_severity)

Reasoning: call the tools in the order that makes sense. You do NOT need to
call all tools if early results make the answer obvious (e.g., no deontic
language → reject immediately).

Return a JSON object with EXACTLY these keys (no extras):
{
  "is_candidate": true | false,
  "clause_type": "obligation" | "prohibition" | "exception" | "exemption" | "definition" | "other",
  "severity": "high" | "medium" | "low",
  "confidence": 0.0–1.0,
  "reason": "one-sentence explanation"
}
"""


class PolicyMiningAgent:
    """
    LLM-assisted clause candidate detector.

    INTEGRATION POINT — step2_clause_miner.py
    ──────────────────────────────────────────
    The existing `mine_candidates()` function assigns a `confidence` score
    (0.05–0.95) based on regex matches. Any span with confidence < 0.6 (the
    "grey zone") can be passed to this agent for a second opinion before being
    dropped, recovering policy clauses that regex misses.

    Example integration in step2_clause_miner.py::

        from agents.langchain_agents import PolicyMiningAgent
        _agent = PolicyMiningAgent()   # instantiate once, reuse

        # inside mine_candidates(), after computing `conf`:
        if conf < 0.6:
            decision = _agent.run(
                text=span["text"],
                section_path=span.get("section_path", []),
                span_id=span.get("span_id", ""),
                context_prev=span.get("context_prev", ""),
                context_next=span.get("context_next", ""),
            )
            if not decision.get("is_candidate"):
                continue   # agent says skip
            # promote with agent's metadata
            cand["type"]       = decision["clause_type"]
            cand["severity"]   = decision["severity"]
            cand["confidence"] = decision["confidence"]
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self._executor = _make_executor(
            _get_llm(model),
            _MINING_TOOLS,
            _MINING_SYSTEM,
        )

    def run(
        self,
        text: str,
        section_path: List[str] | None = None,
        span_id: str = "",
        context_prev: str = "",
        context_next: str = "",
    ) -> Dict[str, Any]:
        """
        Decide whether `text` is a policy candidate clause.

        Parameters
        ----------
        text          : The raw span text.
        section_path  : List of section heading strings (e.g. ["Chapter 2", "Obligations"]).
        span_id       : Unique span identifier for tracing.
        context_prev  : Previous sentence for context.
        context_next  : Following sentence for context.

        Returns
        -------
        dict with keys: is_candidate, clause_type, severity, confidence, reason
        """
        section_path = section_path or []
        user_input = json.dumps({
            "text": text,
            "section_path": section_path,
            "span_id": span_id,
            "context_prev": context_prev,
            "context_next": context_next,
        }, ensure_ascii=False)

        raw = self._executor.invoke({"input": user_input})
        output = raw.get("output", "{}")

        # Parse final JSON from agent output
        try:
            # Agent may wrap JSON in markdown fences
            json_match = re.search(r"\{.*\}", output, re.DOTALL)
            result = json.loads(json_match.group(0)) if json_match else {}
        except Exception:
            result = {}

        return {
            "is_candidate": bool(result.get("is_candidate", False)),
            "clause_type":  result.get("clause_type", "other"),
            "severity":     result.get("severity", "medium"),
            "confidence":   float(result.get("confidence", 0.5)),
            "reason":       result.get("reason", ""),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT 2 — TestabilityJudgeAgent
#  WHERE:  Replaces the _chat_json() call inside tag_testability() in
#          step5_llm_as_judge_testability.py.
#  WHY:    The current single-shot prompt compresses the A/B/C decision into
#          one LLM call. An agent that calls separate tools per criterion
#          (actionability → boundedness → observability) creates an inspectable
#          reasoning chain, reduces hallucinated evidence signals, and lets us
#          catch the common failure mode where a rule is actionable but has
#          NO observable oracle (→ should be testable=False).
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def check_actionability(clause_text: str) -> str:
    """
    CRITERION A — Actionability.
    Decide whether the clause contains an obligation or prohibition
    (must / shall / may not / required / prohibited / forbid / …).
    Returns: {"actionable": bool, "matched_tokens": [...]}.
    A clause with no obligation/prohibition language is NOT testable.
    """
    _deontic = re.compile(
        r"\b(shall(?:\s+not)?|must(?:\s+not)?|may\s+not|may\s+only|"
        r"required?\s+to|prohibited?\s+from|forbidden|prohibited|"
        r"ensure\s+that|verify\s+that|obliged?\s+to|mandated?\s+to)\b",
        re.IGNORECASE,
    )
    matches = [m.group(0) for m in _deontic.finditer(clause_text)]
    return json.dumps({"actionable": bool(matches), "matched_tokens": matches})


@tool
def check_boundedness(clause_text: str) -> str:
    """
    CRITERION B — Boundedness.
    Decide whether the clause names an explicit or clearly implied
    subject (who), object (what), or condition (when/where).
    Returns: {"bounded": bool, "reason": str}.
    Pure principles ("AI should be fair") without a concrete subject fail here.
    """
    actor_hint = re.search(
        r"\b(provider|deployer|user|org|model|agent|developer|admin|"
        r"operator|controller|processor|organization|system|service)\b",
        clause_text, re.IGNORECASE,
    )
    data_hint = re.search(
        r"\b(PHI|PII|data|log|record|request|response|prompt|output|"
        r"token|model|config|credential|secret|file|report)\b",
        clause_text, re.IGNORECASE,
    )
    condition_hint = re.search(
        r"\b(when|where|if|upon|before|after|within|during|at\s+least|"
        r"no\s+later\s+than|always|every)\b",
        clause_text, re.IGNORECASE,
    )
    bounded = bool(actor_hint or data_hint or condition_hint)
    reason = (
        f"Actor='{actor_hint.group(0) if actor_hint else None}' "
        f"Data='{data_hint.group(0) if data_hint else None}' "
        f"Condition='{condition_hint.group(0) if condition_hint else None}'"
    )
    return json.dumps({"bounded": bounded, "reason": reason})


@tool
def identify_evidence_signals(clause_text: str) -> str:
    """
    CRITERION C — Observability (part 1).
    Map the clause to zero or more evidence signal types:
      io_check   : inspect app/model I/O (prompts, responses, refusals, redactions)
      log_check  : inspect runtime logs / telemetry
      config_check: inspect static config / YAML / JSON policies
      ci_gate    : build/deploy gates, risk reports, test outputs
      data_check : datasets/metadata (PII tags, license headers)
      repo_check : code repo state (licenses, PR templates, required files)
    Returns: {"evidence_signals": [list]}.
    An empty list means no verifiable oracle → NOT testable.
    """
    text = clause_text.lower()
    signals = []
    if re.search(r"\b(prompt|response|output|refus|redact|reply|answer|generat)\b", text):
        signals.append("io_check")
    if re.search(r"\b(log|audit|telemetr|trace|event|correlat|monitor|record)\b", text):
        signals.append("log_check")
    if re.search(r"\b(config|policy|setting|yaml|json|rule|flag|parameter|threshold)\b", text):
        signals.append("config_check")
    if re.search(r"\b(ci|cd|pipeline|deploy|build|gate|report|test|scan|lint)\b", text):
        signals.append("ci_gate")
    if re.search(r"\b(dataset|data set|pii\s+tag|license|annotation|metadata|schema)\b", text):
        signals.append("data_check")
    if re.search(r"\b(repo|git|pull\s+request|merge\s+request|code\s+review|branch|commit)\b", text):
        signals.append("repo_check")
    return json.dumps({"evidence_signals": signals})


@tool
def check_observability(signals_json: str) -> str:
    """
    CRITERION C — Observability (part 2).
    Given a JSON array of evidence signals (from identify_evidence_signals),
    confirm at least one valid oracle exists for pass/fail verification.
    Returns: {"observable": bool, "reason": str}.
    Call this after identify_evidence_signals.
    """
    try:
        signals = json.loads(signals_json)
        if not isinstance(signals, list):
            signals = []
    except Exception:
        signals = []

    valid = [s for s in signals if s in EVIDENCE_SIGNALS]
    observable = bool(valid)
    reason = (
        f"Valid oracles: {valid}" if observable
        else "No evidence signal maps to a verifiable artifact."
    )
    return json.dumps({"observable": observable, "reason": reason})


_JUDGE_TOOLS = [
    check_actionability,
    check_boundedness,
    identify_evidence_signals,
    check_observability,
]

_JUDGE_SYSTEM = """\
You are a governance QA judge for an AI policy testing pipeline.
Your task: classify a policy rule as operationally testable or not.

DECISION PROCEDURE — follow these steps in order:
1. Call check_actionability to verify Criterion A.
   If NOT actionable → return testable=false immediately.
2. Call check_boundedness to verify Criterion B.
   If NOT bounded → return testable=false with reason.
3. Call identify_evidence_signals to identify oracles.
4. Call check_observability with the signals from step 3.
   If NOT observable → return testable=false.
5. If A AND B AND C all pass → return testable=true.

IMPORTANT:
- Evidence signals must genuinely apply. Do not add signals speculatively.
- "Principle" rules ("AI should be trustworthy") fail Criterion B.
- Rules about processes with no verifiable artifact (meetings, reviews without logs) fail C.

Return EXACTLY this JSON (no extra keys, no markdown):
{
  "testable": true | false,
  "reason": "≤25 words, concrete",
  "evidence_signals": ["io_check", ...]
}
"""


class TestabilityJudgeAgent:
    """
    Multi-step testability classifier using tool-calling.

    INTEGRATION POINT — step5_llm_as_judge_testability.py
    ──────────────────────────────────────────────────────
    Replace the _chat_json() call inside the for-loop of tag_testability()
    with this agent:

        # BEFORE (existing code):
        out, _usage = _chat_json(client, MODEL, SYSTEM_MSG, user_msg)

        # AFTER (agent-based, opt-in):
        from agents.langchain_agents import TestabilityJudgeAgent
        _judge_agent = TestabilityJudgeAgent()   # instantiate once outside loop

        # inside the loop:
        clause = _build_clause_text(r)
        out = _judge_agent.run(rule=r, clause_text=clause)

    The output dict matches the existing keys used downstream:
        testable, reason, evidence_signals
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self._executor = _make_executor(
            _get_llm(model),
            _JUDGE_TOOLS,
            _JUDGE_SYSTEM,
        )

    def run(
        self,
        rule: Dict[str, Any],
        clause_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classify a rule dict as testable or not.

        Parameters
        ----------
        rule         : Rule dict (any subset of the DSL schema fields).
        clause_text  : Optional pre-built clause text. If None, it is
                       synthesised from rule fields (same logic as Step 5).

        Returns
        -------
        dict with keys: testable (bool), reason (str), evidence_signals (list)
        """
        if clause_text is None:
            clause_text = _build_clause_text_from_rule(rule)

        src = rule.get("source", {}) or {}
        user_input = json.dumps({
            "clause_text": clause_text,
            "doc":      src.get("doc", ""),
            "citation": src.get("citation", ""),
            "span_id":  src.get("span_id", ""),
        }, ensure_ascii=False)

        raw = self._executor.invoke({"input": user_input})
        output = raw.get("output", "{}")

        try:
            json_match = re.search(r"\{.*\}", output, re.DOTALL)
            result = json.loads(json_match.group(0)) if json_match else {}
        except Exception:
            result = {}

        signals = result.get("evidence_signals") or []
        if not isinstance(signals, list):
            signals = []
        valid_signals = [s for s in signals if s in EVIDENCE_SIGNALS]

        return {
            "testable":         bool(result.get("testable", False)),
            "reason":           str(result.get("reason", ""))[:200],
            "evidence_signals": valid_signals,
        }


def _build_clause_text_from_rule(rule: dict) -> str:
    """Mirror of step5_llm_as_judge_testability._build_clause_text()."""
    for k in ("source_text", "clause", "text", "original_text"):
        v = rule.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    req = " | ".join(rule.get("requirements", []) or [])
    exc = " | ".join(rule.get("exceptions", []) or [])
    haz = rule.get("hazard", "") or ""
    sc = rule.get("scope", {}) or {}
    actors  = ",".join(sc.get("actor", []) or [])
    domains = ",".join(sc.get("data_domain", []) or [])
    ctx     = ",".join(sc.get("context", []) or [])
    parts = []
    if req:    parts.append(f"Requirement(s): {req}")
    if exc:    parts.append(f"Exception(s): {exc}")
    if haz:    parts.append(f"Hazard: {haz}")
    if any([actors, domains, ctx]):
        parts.append(f"Scope: actor[{actors}] data[{domains}] context[{ctx}]")
    return " | ".join(parts) or "(no clause text available)"


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT 3 — SchemaRepairAgent
#  WHERE:  Called from step3_llm_generation_few_shot.py after a rule fails
#          jsonschema validation, replacing the existing ad-hoc repair logic.
#  WHY:    Step 3 currently does string-level mutations (coerce types, strip
#          bad values) without knowing which specific validation error was
#          triggered. The SchemaRepairAgent first calls validate_rule to get
#          the exact error path, then calls the matching fix tool, then
#          re-validates. This eliminates blind repair attempts and produces a
#          structured audit trail of what was fixed and why.
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def validate_rule(rule_json: str) -> str:
    """
    Validate a rule JSON string against the pipeline's RULE_SCHEMA.
    Returns: {"valid": bool, "errors": [list of error messages]}.
    Always call this first (and again after each fix) to know exactly
    what is broken before attempting a repair.
    """
    try:
        rule = json.loads(rule_json)
    except json.JSONDecodeError as e:
        return json.dumps({"valid": False, "errors": [f"JSON parse error: {e}"]})

    if not _SCHEMA_AVAILABLE or _RULE_VALIDATOR is None:
        # Lightweight fallback: check required fields only
        required = [
            "rule_id", "source", "scope", "hazard", "conditions",
            "exceptions", "requirements", "evidence", "severity",
        ]
        missing = [f for f in required if f not in rule]
        errors = [f"Missing required field: '{f}'" for f in missing]

        # Check severity enum
        sev = rule.get("severity", "")
        if sev and sev.lower() not in SEVERITY_LEVELS:
            errors.append(f"severity '{sev}' not in {sorted(SEVERITY_LEVELS)}")

        # Check arrays
        for arr_field in ("conditions", "exceptions", "requirements", "evidence"):
            v = rule.get(arr_field)
            if v is not None and not isinstance(v, list):
                errors.append(f"'{arr_field}' must be an array, got {type(v).__name__}")

        for scope_field in ("actor", "data_domain", "context"):
            scope = rule.get("scope", {}) or {}
            v = scope.get(scope_field)
            if v is not None and not isinstance(v, list):
                errors.append(f"scope.{scope_field} must be an array, got {type(v).__name__}")

        return json.dumps({"valid": not errors, "errors": errors})

    # Full jsonschema validation
    errors = [
        f"[{'/'.join(str(p) for p in e.absolute_path)}] {e.message}"
        for e in _RULE_VALIDATOR.iter_errors(rule)
    ]
    return json.dumps({"valid": not errors, "errors": errors})


@tool
def fix_severity(requirements_text: str, hazard_text: str) -> str:
    """
    Infer the correct severity enum value ("low"|"medium"|"high"|"critical")
    from the requirements and hazard text.
    Returns: {"severity": str}.
    Use this when validate_rule reports a severity error.
    """
    combined = (requirements_text + " " + hazard_text).lower()
    if re.search(r"\b(critical|fatal|breach|exfiltr|unlawful|gdpr\s+fine)\b", combined):
        sev = "critical"
    elif re.search(r"\b(MUST|SHALL|PROHIBITED|UNLAWFUL)\b", requirements_text):
        sev = "high"
    elif re.search(r"\b(shall|must|prohibit|required|forbidden)\b", combined, re.I):
        sev = "high"
    elif re.search(r"\b(should|recommend|prefer)\b", combined, re.I):
        sev = "medium"
    else:
        sev = "low"
    return json.dumps({"severity": sev})


@tool
def coerce_to_array(value_json: str) -> str:
    """
    Coerce a scalar value (string, number, null) to a single-element array,
    or return the value unchanged if it is already an array.
    Input:  JSON-encoded value (e.g. '"provider"' or '["provider"]' or 'null').
    Returns: {"result": [list]}.
    Use this when validate_rule reports that a field must be an array.
    """
    try:
        val = json.loads(value_json)
    except Exception:
        val = value_json  # treat as raw string
    if val is None:
        result = []
    elif isinstance(val, list):
        result = val
    else:
        result = [val]
    return json.dumps({"result": result})


@tool
def canonicalize_scope_fields(scope_json: str) -> str:
    """
    Normalize scope.actor, scope.data_domain, scope.context values to the
    canonical enumerations used by the pipeline schema (ACTOR_ENUM, DOMAIN_ENUM,
    CONTEXT_ENUM). Unknown values are dropped.
    Input:  JSON-encoded scope dict.
    Returns: {"scope": {actor:[...], data_domain:[...], context:[...]}}.
    Use this when validate_rule reports enum violations in scope fields.
    """
    try:
        scope = json.loads(scope_json)
        if not isinstance(scope, dict):
            scope = {}
    except Exception:
        scope = {}

    if _SCHEMA_AVAILABLE:
        dummy_rule = {"scope": scope}
        canonicalize_scope(dummy_rule)
        return json.dumps({"scope": dummy_rule["scope"]})

    # Lightweight fallback: coerce to list, lowercase, keep known tokens
    _known_actors  = {"user","model","agent","org","provider","deployer","developer",
                      "evaluator","admin","auditor","security","legal","privacy","risk"}
    _known_domains = {"PHI","PII","health","code","text","image","audio","general",
                      "financial","credentials","secrets","telemetry","document"}
    _known_contexts= {"prod","staging","eval","tenant","repo","high-risk","public",
                      "internal","research","training","finetune","dev","ci","qa"}

    def _clean(raw, known):
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        return [v for v in raw if v in known]

    return json.dumps({"scope": {
        "actor":       _clean(scope.get("actor",       []), _known_actors),
        "data_domain": _clean(scope.get("data_domain", []), _known_domains),
        "context":     _clean(scope.get("context",     []), _known_contexts),
    }})


@tool
def fill_missing_required_fields(rule_json: str) -> str:
    """
    Add empty-but-valid defaults for any missing required fields so the rule
    at least passes structural validation (further content repair can follow).
    Required fields: rule_id, source(doc,citation,span_id), scope(actor,
    data_domain,context), hazard, conditions, exceptions, requirements,
    evidence, severity.
    Returns: {"rule": <patched rule dict>}.
    Use this when validate_rule reports missing required fields.
    """
    try:
        rule = json.loads(rule_json)
        if not isinstance(rule, dict):
            rule = {}
    except Exception:
        rule = {}

    rule.setdefault("rule_id", None)
    src = rule.setdefault("source", {})
    src.setdefault("doc", "")
    src.setdefault("citation", "")
    src.setdefault("span_id", "")
    sc = rule.setdefault("scope", {})
    sc.setdefault("actor",       [])
    sc.setdefault("data_domain", [])
    sc.setdefault("context",     [])
    rule.setdefault("hazard",        "")
    rule.setdefault("conditions",    [])
    rule.setdefault("exceptions",    [])
    rule.setdefault("requirements",  [])
    rule.setdefault("evidence",      [])
    rule.setdefault("severity",      "medium")

    return json.dumps({"rule": rule})


_REPAIR_TOOLS = [
    validate_rule,
    fix_severity,
    coerce_to_array,
    canonicalize_scope_fields,
    fill_missing_required_fields,
]

_REPAIR_SYSTEM = """\
You are a JSON schema repair agent for an AI policy extraction pipeline.
You receive a rule dict that failed schema validation and must repair it
until it is fully valid.

REPAIR PROCEDURE:
1. Call validate_rule with the current rule JSON to get the exact error list.
2. For each error, call the appropriate fix tool:
   - Missing required fields       → fill_missing_required_fields
   - severity value wrong case/value → fix_severity (pass requirements + hazard text)
   - scope.* must be array         → coerce_to_array, then update scope field
   - scope enum violations         → canonicalize_scope_fields
   - other array field not a list  → coerce_to_array for that field
3. After all fixes, call validate_rule again to confirm.
4. If still invalid after 3 repair rounds, return the best partial result.

RULES:
- Do NOT invent data that is not in the original rule.
- Do NOT remove fields — only coerce types and normalize enums.
- Preserve all original string content (hazard, requirements text, etc.).

Return EXACTLY this JSON (no markdown):
{
  "repaired_rule": { <full rule dict> },
  "changes": ["description of each change made"],
  "valid": true | false
}
"""


