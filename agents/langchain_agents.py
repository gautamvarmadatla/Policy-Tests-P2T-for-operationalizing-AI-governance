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


