#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, logging
from pathlib import Path
from typing import Optional, Dict, List

# ---------- Logging setup ----------
logger = logging.getLogger("policy_pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("PIPELINE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

# ---------- Step imports ----------
from step1_ingest_pdf import robust_ingest_pdf
from step2_clause_miner import run_clause_miner
from step3_llm_generation_few_shot import run_hardened
from step4_semantic_dedup import run_dedup
from step5_llm_as_judge_testability import tag_testability
from step6_example_gen_iocheck import generate_examples


def _t() -> float:
    return time.perf_counter()

# --- Pass-through candidate writer (bypass Step 2) ---
def write_passthrough_candidates(spans_jsonl: Path, out_jsonl: Path) -> Path:
    """
    Build a 'candidates' JSONL directly from spans.jsonl:
    - Keeps span_id, doc_id (derived from span_id if missing), section_path, page, text
    - Adds empty context_prev/context_next
    - Sets type='other', severity='medium', normative=True, confidence=0.5
    """
    import json
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(spans_jsonl, "r", encoding="utf-8") as fin, \
         open(out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            if str(r.get("kind", "")).lower() == "table":
                continue

            span_id = r.get("span_id", "")
            if not span_id:
                continue
            
            doc_id = r.get("doc_id")
            if not doc_id and isinstance(span_id, str) and "/" in span_id:
                doc_id = span_id.split("/", 1)[0]

            rec = {
                "span_id": span_id,
                "doc_id": doc_id or "",
                "section_path": r.get("section_path") or [],
                "page": r.get("page"),
                "text": r.get("text", ""),
                "context_prev": [],
                "context_next": [],
                "type": "other",
                "severity": "medium",
                "cues": [],
                "cross_refs": [],
                "jurisdictional": False,
                "deadline": None,
                "threshold": None,
                "normative": True,
                "inherited_from": None,
                "confidence": 0.5,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1

    logger.info("Pass-through candidates written: %d → %s", total, out_jsonl)
    return out_jsonl


def run_pipeline(
    *,
    pdf_path: Path,
    out_dir: Optional[Path] = None,
    model_step3: Optional[str] = None,
    model_step5_6: Optional[str] = None,
    enable_images: bool = False,
    enable_tables: bool = False,
    run_step2: bool = True, 
    run_step4: bool = True,
    run_step5: bool = True,
    run_step6: bool = True,
    # --- Step 3 pass-through knobs ---
    provider_step3: str = "openai",
    decoder_step3: str = "jsonschema",    # or "grammar"
    evidence_gate: bool = True,
    allow_domains: Optional[List[str]] = None,  # e.g. [".gov", "europa.eu"]
    smt: bool = False,
    repair_passes: int = 1,
    counterfactuals: int = 0,
    # NEW: LLM-based judge
    llm_judge: bool = False,
    llm_judge_model: Optional[str] = None,
    decode_retries: int = 3,  
    spans_jsonl_override: Optional[Path] = None,
    group_n: int = 4,
) -> Dict[str, Optional[str]]:
    """
    End-to-end Steps 1→6. Returns dict of written paths.
    Steps 1–3 always run. Steps 4–6 can be toggled with run_step{4,5,6}.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        logger.error("Input PDF not found: %s", pdf_path)
        raise FileNotFoundError(pdf_path)

    # out root is based on document name by default: <pdf_dir>/out/<doc_stem>/
    doc_stem = pdf_path.stem
    out_root = Path(out_dir).resolve() if out_dir else (pdf_path.parent / "out" / doc_stem)
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Pipeline start | pdf=%s | out_dir=%s", pdf_path, out_root)

    results: Dict[str, Optional[str]] = {}
    t0 = _t()

    # ---------- STEP 1: PDF → JSONL ----------
    step_name = "Step 1: ingest PDF"
    t = _t()
    try:
        logger.info("%s: started (images=%s, tables=%s)", step_name, enable_images, enable_tables)
        spans_jsonl = Path(
            robust_ingest_pdf(
                pdf_path=str(pdf_path),
                out_jsonl=str(out_root / "spans.jsonl"),
                enable_images=enable_images,
                enable_tables=enable_tables,
                out_base_dir=str(out_root),
                group_n=group_n,
            )
        )
        results["step1_spans_jsonl"] = str(spans_jsonl)
        logger.info("%s: done in %.2fs → %s", step_name, _t() - t, spans_jsonl)
    except Exception:
        logger.exception("%s: failed", step_name)
        raise

    # ---------- STEP 2: clause miner (optional) ----------
    step_name = "Step 2: clause miner"
    t = _t()
    try:
        if run_step2:
            logger.info("%s: started", step_name)
            candidates_jsonl = Path(run_clause_miner(str(spans_jsonl)))
            logger.info("%s: done in %.2fs → %s", step_name, _t() - t, candidates_jsonl)
        else:
            logger.info("%s: skipped (pass-through)", step_name)
            # write passthrough candidates next to Step 1
            candidates_jsonl = Path(out_root / "spans_passthrough_candidates.jsonl")
            write_passthrough_candidates(spans_jsonl, candidates_jsonl)
        results["step2_candidates_jsonl"] = str(candidates_jsonl)
    except Exception:
        logger.exception("%s: failed", step_name)
        raise

    # ---------- STEP 3: LLM extraction ----------
    step_name = "Step 3: LLM extraction"
    t = _t()
    schema_yaml = out_root / "policy_schema.yaml"
    per_candidate_jsonl = out_root / f"{pdf_path.stem}.extracted.jsonl"
    try:
        logger.info("%s: started (model=%s)", step_name, model_step3 or "(default)")
        run_hardened(
            candidates_jsonl=str(candidates_jsonl),
            out_schema_path=str(schema_yaml),
            out_rules_jsonl=str(per_candidate_jsonl),
            provider=provider_step3,
            model=model_step3,
            decoder=decoder_step3,
            evidence_gate=evidence_gate,
            allow_domains=allow_domains,
            smt=smt,
            repair_passes=repair_passes,
            counterfactuals=counterfactuals,
            llm_judge=llm_judge,
            llm_judge_model=llm_judge_model,
            decode_retries=decode_retries,
        )
        # normalize if PyYAML missing
        if not schema_yaml.exists():
            alt = schema_yaml.with_suffix(".json")
            if alt.exists():
                schema_yaml = alt
                logger.warning("YAML missing; using JSON schema at %s", alt)
            else:
                raise FileNotFoundError("Step 3 schema not found (.yaml/.json).")

        results["step3_schema"] = str(schema_yaml)
        results["step3_per_candidate_jsonl"] = str(per_candidate_jsonl)
        logger.info(
            "%s: done in %.2fs → schema=%s, per-candidate=%s",
            step_name, _t() - t, schema_yaml, per_candidate_jsonl
        )
    except Exception:
        logger.exception("%s: failed", step_name)
        raise

    # ---------- STEP 4: semantic dedup ----------
    if run_step4:
        step_name = "Step 4: semantic dedup"
        t = _t()
        try:
            logger.info("%s: started", step_name)
            cleaned_path, dups_tsv, spans_tsv, pairs_tsv = run_dedup(
                inputs=[schema_yaml],
                output_path=out_root / "policy_schema.cleaned.yaml",
                strict=False,
                semantic=True,
                threshold=0.90,
                doc_block=True,
                preview=True,
                preview_k=200,
                preview_path=out_root / "policy_schema.cleaned.pairs.tsv",
            )
            if not cleaned_path.exists():
                raise FileNotFoundError(f"Step 4 cleaned not found: {cleaned_path}")

            results["step4_cleaned_rules"] = str(cleaned_path)
            results["step4_dups_tsv"] = str(dups_tsv)
            results["step4_spans_tsv"] = str(spans_tsv)
            results["step4_pairs_tsv"] = str(pairs_tsv) if pairs_tsv else None
            logger.info("%s: done in %.2fs → cleaned=%s", step_name, _t() - t, cleaned_path)
        except Exception:
            logger.exception("%s: failed", step_name)
            raise
    else:
        logger.info("Step 4: skipped by configuration")
        cleaned_path = None
        results["step4_cleaned_rules"] = None
        results["step4_dups_tsv"] = None
        results["step4_spans_tsv"] = None
        results["step4_pairs_tsv"] = None

    # ---------- STEP 5: tag is_testable ----------
    if run_step5:
        if not run_step4 or not cleaned_path:
            raise RuntimeError(
                "Step 5 requires Step 4 outputs. Re-run with --run-step4 (default) "
                "or do not pass --no-step4."
            )
        step_name = "Step 5: tag is_testable"
        t = _t()
        try:
            testable_out = cleaned_path.with_name(
                cleaned_path.stem.replace(".cleaned", "") + ".with_testable"
            ).with_suffix(cleaned_path.suffix)

            if tag_testability is None:
                raise RuntimeError("Step 5 shim `tag_testability()` not found in step5_llm_as_judge_testability.py")

            logger.info("%s: started (model=%s)", step_name, model_step5_6 or "(default)")
            testable_written = tag_testability(cleaned_path, testable_out, model_step5_6)
            results["step5_with_testable"] = str(testable_written)
            logger.info("%s: done in %.2fs → %s", step_name, _t() - t, testable_written)
        except Exception:
            logger.exception("%s: failed", step_name)
            raise
    else:
        logger.info("Step 5: skipped by configuration")
        testable_written = None
        results["step5_with_testable"] = None

    # ---------- STEP 6: generate io_check examples ----------
    if run_step6:
        if not run_step5 or not testable_written:
            raise RuntimeError(
                "Step 6 requires Step 5 outputs. Re-run with --run-step5 (default) "
                "or do not pass --no-step5."
            )
        step_name = "Step 6: io_check examples"
        t = _t()
        try:
            if generate_examples is None:
                raise RuntimeError("Step 6 shim `generate_examples()` not found in step6_example_gen_iocheck.py")

            logger.info("%s: started (model=%s)", step_name, model_step5_6 or "(default)")
            examples_written = generate_examples(Path(results["step5_with_testable"]), model_step5_6, target_n=5)
            results["step6_examples_yaml"] = str(examples_written)
            logger.info("%s: done in %.2fs → %s", step_name, _t() - t, examples_written)
        except Exception:
            logger.exception("%s: failed", step_name)
            raise
    else:
        logger.info("Step 6: skipped by configuration")
        results["step6_examples_yaml"] = None

    logger.info("Pipeline finished in %.2fs", _t() - t0)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("End-to-end policy pipeline (Steps 1→6).")
    ap.add_argument("--pdf", required=True, help="Input PDF path.")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <pdf>/out/<doc_stem>)")
    ap.add_argument("--model-step3", default=None, help="Model for Step 3 extraction.")
    ap.add_argument("--model-step5-6", default=None, help="Model for Steps 5/6 (judge + examples).")
    ap.add_argument("--no-images", action="store_true", help="Disable Step 1 image/caption processing.")
    ap.add_argument("--no-tables", action="store_true", help="Disable Step 1 table extraction.")
    ap.add_argument("--no-step4", action="store_true", help="Skip Step 4 (semantic dedup).")
    ap.add_argument("--no-step5", action="store_true", help="Skip Step 5 (testability tagging).")
    ap.add_argument("--no-step6", action="store_true", help="Skip Step 6 (io_check examples).")
    ap.add_argument("--provider-step3", default="openai",
                    choices=["openai","gemini"], help="Provider for Step 3.")
    ap.add_argument("--decoder-step3", default="jsonschema",
                    choices=["jsonschema","grammar"], help="Decoder for Step 3.")
    ap.add_argument("--no-evidence-gate", action="store_true",
                    help="Disable evidence gate in Step 3.")
    ap.add_argument("--allow-domain", dest="allow_domains", nargs="*",
                    help="Allow-list of domains for evidence URLs (e.g. .gov europa.eu).")
    ap.add_argument("--smt", action="store_true",
                    help="Enable SMT conflict check in Step 3.")
    ap.add_argument("--repair-passes", type=int, default=1,
                    help="Repair passes for Step 3.")
    ap.add_argument("--counterfactuals", type=int, default=0,
                    help="Counterfactual probes for Step 3.")
    # NEW CLI switches for LLM judge
    ap.add_argument("--llm-judge", action="store_true",
                    help="Enable LLM-based judging in Step 3.")
    ap.add_argument("--llm-judge-model", default=None,
                    help="Override model for LLM judge (default: Step 3 model).")
    ap.add_argument("--decode-retries", type=int, default=3,
                help="Max decode/validation retries per span (repair-on-parse).")
    ap.add_argument("--spans-jsonl", dest="spans_jsonl_override", default=None,
                    help="Optional path to an existing spans.jsonl (otherwise Step 1 output is used).")
    ap.add_argument("--group-n", type=int, default=4,
                    help="Group this many sentences per emitted span in Step 1 (default: 4).")
    ap.add_argument("--no-step2", action="store_true", help="Skip Step 2 (clause miner) and pass spans directly to Step 3.")
    args = ap.parse_args()

    try:
        results = run_pipeline(
            pdf_path=Path(args.pdf),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            model_step3=args.model_step3,
            model_step5_6=args.model_step5_6,
            enable_images=not args.no_images,
            enable_tables=not args.no_tables,
            run_step2=not args.no_step2, 
            run_step4=not args.no_step4,
            run_step5=not args.no_step5,
            run_step6=not args.no_step6,
            provider_step3=args.provider_step3,
            decoder_step3=args.decoder_step3,
            evidence_gate=not args.no_evidence_gate,
            allow_domains=args.allow_domains,
            smt=args.smt,
            repair_passes=args.repair_passes,
            counterfactuals=args.counterfactuals,
            llm_judge=args.llm_judge,
            llm_judge_model=args.llm_judge_model,
            decode_retries=args.decode_retries,
            spans_jsonl_override=Path(args.spans_jsonl_override) if args.spans_jsonl_override else None,
            group_n=args.group_n,
        )
        print(json.dumps(results, indent=2))
    except Exception:
        logger.exception("Pipeline aborted due to an error.")
        raise
