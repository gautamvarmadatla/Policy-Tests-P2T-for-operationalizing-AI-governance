# Executable Governance for AI

Utilities and research code for generating, curating, and testing executable governance policies for AI systems. The workflow turns policy text (e.g., PDFs) into structured rules, tags them for testability, generates example prompts, and produces runnable tests or guardrails integrations.

This README covers setup, running the end‑to‑end pipeline, how outputs are organized, and how to integrate rules with a lightweight rails engine.

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Environment & Config](#environment--config)
- [Repository Layout](#repository-layout)
- [Pipeline Workflow](#pipeline-workflow)
- [CLI Usage](#cli-usage)
- [Manual Usage (Per Step)](#manual-usage-per-step)
- [Policy Tests Output](#policy-tests-output)
- [Rails Integration](#rails-integration)
- [Annotator Data & Golden Dataset](#annotator-data--golden-dataset)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Quick Start
- Create a virtual environment and install dependencies.
- Set `OPENAI_API_KEY` for LLM calls used in Steps 3–6 (and optional Step 4 semantic dedup).
- Run the pipeline on a PDF and review outputs.

PowerShell (Windows):
- `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1`
- `python -m pip install --upgrade pip`
- `pip install -r requirements.txt`
- `setx OPENAI_API_KEY "<your_api_key>"`
- `python cli.py pipeline .\\path\\to\\policy.pdf --enable-tables`

macOS/Linux:
- `python3 -m venv .venv && source .venv/bin/activate`
- `python -m pip install --upgrade pip`
- `pip install -r requirements.txt`
- `export OPENAI_API_KEY="<your_api_key>"`
- `python cli.py pipeline path/to/policy.pdf --enable-tables`

## Installation
- Python 3.8+ recommended.
- Install all project dependencies:
  - `pip install -r requirements.txt`
- Optional extras used by certain steps:
  - `nltk` sentence tokenizer: if you enable NLTK splitting in Step 1, run `python -c "import nltk; nltk.download('punkt')"` once.

## Environment & Config
- `OPENAI_API_KEY`: required for Steps 3, 5, 6, and for Step 4 when semantic dedup is enabled.
- `POLICY_JUDGE_MODEL`: model name for LLM-as-judge (defaults to `gpt-4o-mini`).
- `AGENT_BASE_URL`: base URL for your app under test (used by `judge_helpers.py`, default `http://127.0.0.1:8000`).
- `POLICY_RESULTS_PATH`: file path for accumulating test results (default `policy_results.jsonl`).

## Repository Layout
- `master.py` – Orchestrates Steps 1–6 programmatically. ( Master file to use ) 
- `step1_ingest_pdf.py` – PDF → JSONL span extraction (sentences, captions, tables).
- `step2_clause_miner.py` – Heuristics to mine clause candidates from spans.
- `step3_llm_generation_few_shot.py` – LLM extraction of structured rules from candidates.
- `step4_semantic_dedup.py` – Structural and optional semantic deduplication of rules.
- `step5_llm_as_judge_testability.py` – Tag rules for operational testability via LLM.
- `step6_example_gen_iocheck.py` – Generate benign/adversarial prompts for IO‑checkable rules.
- `generate_llmjudge_tests_nemo.py` – Example generator for pytest tests (NEMO style; narrow demo selection).
- `judge_helpers.py` – Calls your app under test and runs an LLM judge; appends JSONL results.
- `tools/rules_to_nemo.py` – Convert cleaned rules to a `rails/actions/policy_rules.json` table.
- `rails/` – Lightweight rails integration (NeMo Guardrails actions and config/DSL).
- `Annotator Data/` – Annotation CSVs, JSONL spans, and notebook assets.

## Pipeline Workflow

<img width="1368" height="515" alt="image" src="https://github.com/user-attachments/assets/ea0b4d33-9c0d-48cc-af89-5ca6d200da4e" />



### 1) Ingest PDF → spans JSONL
- Groups sentences, extracts captions/tables, de‑noises headers/footers.

### 2) Mine clause candidates
- Heuristics oriented to obligations/prohibitions/exceptions; adds context and metadata.

### 3) Extract structured rules with LLM ( Generation + Judge + repair + SMT + Evidence gate + Counterfactuals )
- Normalizes to a policy schema; can enforce JSON schema; optional SMT consistency checks.

### 4) Deduplicate rules
- Structural signature dedup always; optional semantic dedup with embeddings.

### 5) Tag testability with LLM
- Decides if a rule is operationally testable; records signals like `io_check`.

### 6) Generate example prompts
- Creates benign/adversarial examples for IO‑checkable rules.

Outputs are written near the input PDF under `out/<pdf_stem>/`.

## CLI Usage
- End‑to‑end run on a single PDF:
  - `python cli.py pipeline <path_to_pdf> [--out-dir OUT] [--enable-tables] [--enable-images] [--no-step{2,4,5,6}] [--model-step3 NAME] [--model-step5-6 NAME] [--group-n N]`

Examples:
- `python cli.py pipeline docs\\EU_AI_Act.pdf --enable-tables`
- `python cli.py pipeline ./WHO.pdf --no-step4 --model-step3 gpt-4o-mini`

Programmatic entry: see `master.py:run_pipeline`.

## Manual Usage (Per Step)
If you prefer running individual steps or customizing behavior:
- Step 1 (import): `from step1_ingest_pdf import robust_ingest_pdf`
- Step 2: `from step2_clause_miner import run_clause_miner`
- Step 3: `from step3_llm_generation_few_shot import run_hardened`
- Step 4: `from step4_semantic_dedup import run_dedup`
- Step 5: `from step5_llm_as_judge_testability import tag_testability`
- Step 6: `from step6_example_gen_iocheck import generate_examples`

Notes
- Some scripts also expose CLI examples internally; inspect the file tops for arguments if you prefer direct CLI.
- Step 4 semantic dedup uses OpenAI embeddings (`text-embedding-3-large`) when enabled.


Generating Tests (Demo)
- `python generate_llmjudge_tests_nemo.py <rules.yaml> policy_tests_out`
- Note: the current demo filter is intentionally narrow and may generate zero tests unless a matching rule exists.

## Rails Integration
- Convert rules to a compact JSON table for rails actions:
  - `python tools\\rules_to_nemo.py <rules.cleaned.yaml> rails\\actions\\policy_rules.json`
- NeMo Guardrails actions live in `rails/actions/policy_actions.py` and rely on `policy_rules.json`.
- The default `generate_bot_message` action is a stub; integrate with your own LLM/runtime while keeping the judge/revise helpers if desired.

Key Env Vars
- `OPENAI_API_KEY`, `POLICY_JUDGE_MODEL`, `AGENT_BASE_URL`.

## Annotator Data & Golden Dataset
- `Annotator Data/annotator_csvs/` and `Annotator Data/Annotator Docs/out/` contain CSVs and JSONL spans for multiple sources (WHO, EU, HIPAA, Microsoft, NIST).
- `golden_dataset.py` builds a gold CSV from a JSONL of candidates (no filtering/dedup). Adjust paths inside before running.

## Troubleshooting
- Missing `OPENAI_API_KEY`:
  - Set the variable in your shell and restart the session so libraries can read it.
- NLTK not found or tokenizer error:
  - Run `python -c "import nltk; nltk.download('punkt')"` or rely on the built‑in regex splitter.
- PDF tables not extracted:
  - Ensure `pdfplumber` is installed and pass `--enable-tables`.
- No tests generated by the demo script:
  - The generator is intentionally selective (IO‑check only and specific demo rule). Use rails or extend the generator for broader coverage.

## Contributing
- Use branches and pull requests. Add tests for behavior changes.
- Keep exploratory work in notebooks, promote stable logic into modules.
- If you add new dependencies, update `requirements.txt`.

## License
- No license file is included. Add a `LICENSE` if you plan to open‑source this project. I can recommend one based on your goals.

## Example Output
After running `python cli.py pipeline path/to/policy.pdf --enable-tables`, you should see a structure like:

- out/<policy_stem>/
  - spans.jsonl
  - spans_passthrough_candidates.jsonl (if Step 2 skipped)
  - policy_schema.yaml or .json
  - <policy_stem>.extracted.jsonl
  - policy_schema.cleaned.yaml (.json if PyYAML unavailable)
  - policy_schema.cleaned.yaml.dups.tsv
  - policy_schema.cleaned.yaml.spans.tsv
  - (optional) semantic_pairs_preview.tsv

Generated demo tests (if used):
- policy_tests_out/
  - policy_results.jsonl
  - tests/
    - test_<rule_id>.py
