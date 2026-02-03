# Tag each rule with is_testable using GPT-5 (JSON-only, no assumptions/confidence in output)
# Prereqs: pip install openai pyyaml
# Env: set OPENAI_API_KEY

import os, json, time, math, random, datetime, textwrap
from pathlib import Path

import yaml
from openai import OpenAI
from openai._exceptions import OpenAIError, RateLimitError, APIStatusError

# --------- CONFIG ----------
INPUT_PATH  = Path(r"D:\AAAI ( AIGOV 26 )\out policy_schema.cleaned.yaml")
OUTPUT_PATH = Path(r"D:\AAAI ( AIGOV 26 )\out policy_schema.with_testable.yaml")
MODEL = "gpt-5-mini"         # your choice here; keep as-is or switch to "gpt-5" / "gpt-5-thinking"
SLEEP_BETWEEN = 0            # keep your current pacing
MAX_RETRIES = 6
LOG_EVERY = 10               # print a running summary every N rules
SHOW_JSON_SNIPPET = True     # print first ~200 chars of raw JSON reply

# --------- PROMPT (no assumptions/confidence) ----------
SYSTEM_MSG = """You are a governance QA judge. Decide whether a policy clause is operationally testable for AI/LLM applications.

Testable only if it states an obligation/prohibition that can be checked with an objective pass/fail oracle using ≥1 of:
- io_check: inspect app/model I/O (prompts, responses, refusals, citations, redactions)
- log_check: inspect runtime logs/telemetry (fields, event types, correlation IDs)
- config_check: inspect static config/policies (YAML/JSON, infra settings, access rules)
- ci_gate: build/deploy gates or reports (risk_report.json thresholds, unit/pytest outputs)
- data_check: datasets/metadata (PII tags, license headers)
- repo_check: code/repo state (licenses, PR templates, required files)

Not testable: pure definitions; principles/mission; scope-only; example-only; processes with no verifiable artifact; vague outcomes without observable predicate.

Decision rule (must all hold):
A) Actionable: obligation/prohibition present (must/shall/may not/require/prohibit…)
B) Bounded: target/condition/subject explicit or reasonably implied (who/what/when/where)
C) Observable: at least one oracle above can verify pass/fail without subjective judgment
If any A/B/C fails → testable=false. If unsure, prefer testable=false and state the missing oracle.

Output JSON only (exact keys; no extras):
{
  "testable": true,
  "reason": "≤25 words, concrete",
  "evidence_signals": ["io_check"]
}
- evidence_signals ∈ {io_check,log_check,config_check,ci_gate,data_check,repo_check}
- Do not invent facts, citations, or artifacts not implied by the clause.
"""

USER_TPL = """Classify the clause below for operational testability.

Clause:
\"\"\"
{CLAUSE_TEXT}
\"\"\"

Optional context: doc="{DOC}", citation="{CIT}", span_id="{SPAN_ID}"

Return only the JSON object.
"""

def _build_clause_text(rule: dict) -> str:
    # Prefer original text if present
    for k in ("source_text","clause","text","original_text"):
        v = rule.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Fallback: synthesize from fields so the model has context
    req = " | ".join(rule.get("requirements", []) or [])
    exc = " | ".join(rule.get("exceptions", []) or [])
    haz = rule.get("hazard", "") or ""
    sc  = rule.get("scope", {}) or {}
    actors  = ",".join(sc.get("actor", []) or [])
    domains = ",".join(sc.get("data_domain", []) or [])
    ctx     = ",".join(sc.get("context", []) or [])
    parts = []
    if req: parts.append(f"Requirement(s): {req}")
    if exc: parts.append(f"Exception(s): {exc}")
    if haz: parts.append(f"Hazard: {haz}")
    if any([actors, domains, ctx]):
        parts.append(f"Scope: actor[{actors}] data[{domains}] context[{ctx}]")
    return " | ".join(parts) or "(no clause text available)"

def _chat_json(client: OpenAI, model: str, system: str, user: str) -> tuple[dict, dict]:
    """
    Robust call with retries + enforced JSON.
    Returns (parsed_json, usage_dict_or_empty).
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content": system},
                          {"role":"user","content": user}],
                response_format={"type":"json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            usage = getattr(resp, "usage", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            } if usage else {}
            return json.loads(content), usage_dict
        except (RateLimitError, APIStatusError) as e:
            last_err = e
            delay = min(8.0, 0.6 * (2 ** attempt)) + random.uniform(0, 0.2)
            print(f"   [rate/HTTP] attempt {attempt+1}/{MAX_RETRIES}, sleeping {delay:.2f}s: {type(e).__name__}: {e}")
            time.sleep(delay)
        except (OpenAIError, json.JSONDecodeError, Exception) as e:
            last_err = e
            delay = 0.5 + 0.2 * attempt
            print(f"   [transient] attempt {attempt+1}/{MAX_RETRIES}, sleeping {delay:.2f}s: {type(e).__name__}: {e}")
            time.sleep(delay)
    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


def tag_testability(input_yaml: Path, output_yaml: Path, model: str | None = None,
                    use_agent: bool = False) -> Path:
    """
    Programmatic entrypoint for Step 5.
    - input_yaml: path to cleaned rules (YAML or JSON with top-level list or {"rules":[...]})
    - output_yaml: path to write tagged rules
    - model: optional override for MODEL
    - use_agent: when True, use TestabilityJudgeAgent (multi-step A→B→C tool calls)
                 instead of the single-shot _chat_json() call.  The agent walks the
                 three testability criteria with separate tools, producing an inspectable
                 reasoning chain.  Requires: pip install langchain langchain-openai.
                 Default: False (original single-shot behaviour).
    Returns the output_yaml path.
    """
    # use your globals but allow overrides
    global INPUT_PATH, OUTPUT_PATH, MODEL
    INPUT_PATH = Path(input_yaml)
    OUTPUT_PATH = Path(output_yaml)
    if model:
        MODEL = model

    # ---- Load YAML/JSON (same as your main flow) ----
    data_in = yaml.safe_load(INPUT_PATH.read_text(encoding="utf-8"))
    rules = data_in.get("rules") if isinstance(data_in, dict) else data_in
    if not isinstance(rules, list):
        raise ValueError("Input must be a list of rule objects or a dict with key 'rules'.")

    # ---- OpenAI client (only needed for single-shot path) ----
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    client = OpenAI(api_key=api_key)

    # ── TestabilityJudgeAgent: init once outside the loop ────────────────────
    # The agent replaces the single _chat_json() call with four sequential tool
    # calls (check_actionability → check_boundedness → identify_evidence_signals
    # → check_observability), making the A/B/C decision criteria explicit and
    # producing an inspectable reasoning chain per rule.
    _judge_agent = None
    if use_agent:
        try:
            from agents.langchain_agents import TestabilityJudgeAgent
            _judge_agent = TestabilityJudgeAgent(model=MODEL)
            print(f"[Step5] TestabilityJudgeAgent enabled (model={MODEL})")
        except Exception as _e:
            import warnings
            warnings.warn(
                f"[Step5] TestabilityJudgeAgent unavailable ({_e}); "
                "falling back to single-shot LLM call.",
                stacklevel=2,
            )
    # ─────────────────────────────────────────────────────────────────────────

    # ---- Process rules (reuse your helpers) ----
    n = len(rules)
    failures = 0
    testable_true = 0
    testable_false = 0
    testable_null = 0

    def _short(s, n=160):
        s = s or ""
        s = " ".join(s.split())
        return (s[:n] + "…") if len(s) > n else s

    t0 = time.time()
    updated = []
    for idx, r in enumerate(rules, start=1):
        src = r.get("source", {}) or {}
        clause = _build_clause_text(r)

        try:
            if _judge_agent is not None:
                # ── Agent path: multi-step tool-calling judgment ──────────────
                out = _judge_agent.run(rule=r, clause_text=clause)
                testable = bool(out.get("testable", False))
                reason   = str(out.get("reason", "")).strip()[:200]
                signals  = out.get("evidence_signals") or []
                # ─────────────────────────────────────────────────────────────
            else:
                # ── Original path: single-shot LLM call ──────────────────────
                user_msg = USER_TPL.format(
                    CLAUSE_TEXT=clause,
                    DOC=(src.get("doc") or ""),
                    CIT=(src.get("citation") or ""),
                    SPAN_ID=(src.get("span_id") or "")
                )
                out, _usage = _chat_json(client, MODEL, SYSTEM_MSG, user_msg)
                testable = bool(out.get("testable", False))
                reason   = str(out.get("reason", "")).strip()[:200]
                signals  = out.get("evidence_signals") or []
                # ─────────────────────────────────────────────────────────────

            if not isinstance(signals, list):
                signals = []

            r["is_testable"] = testable
            r["testability"] = {"reason": reason, "evidence_signals": signals}

            if testable is True:
                testable_true += 1
            elif testable is False:
                testable_false += 1
            else:
                testable_null += 1

        except Exception as e:
            failures += 1
            r["is_testable"] = None
            r["testability"] = {
                "error": f"{type(e).__name__}: {e}",
                "reason": "",
                "evidence_signals": []
            }

        updated.append(r)
        time.sleep(SLEEP_BETWEEN)

    # ---- Write output (preserve top-level shape) ----
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {"rules": updated} if isinstance(data_in, dict) else updated
    OUTPUT_PATH.write_text(
        yaml.safe_dump(out_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    # quick return for master runner
    return OUTPUT_PATH


# Optional: keep CLI usage for standalone runs
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Step 5 — tag rules with is_testable")
    ap.add_argument("--in", dest="inp", required=False, help="Input YAML (cleaned rules)")
    ap.add_argument("--out", dest="out", required=False, help="Output YAML (with testable tags)")
    ap.add_argument("--model", dest="model", required=False, help="Model name override")
    ap.add_argument("--agent", action="store_true", default=False,
                    help="Use TestabilityJudgeAgent (multi-step A/B/C tool calls) instead of single-shot LLM.")
    args = ap.parse_args()
    inp = Path(args.inp) if args.inp else INPUT_PATH
    out = Path(args.out) if args.out else OUTPUT_PATH
    tag_testability(inp, out, args.model, use_agent=args.agent)