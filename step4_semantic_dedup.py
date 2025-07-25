#!/usr/bin/env python3
"""
dedup_policy_rules.py (import-friendly)
End-to-end de-duplication for policy rules (structural + optional semantic).

Expose a single entrypoint:
    run_dedup(inputs, output_path=None, strict=False, semantic=True,
              threshold=0.90, doc_block=True, preview=False, preview_k=200,
              preview_path=None) -> tuple[Path, Path, Path, Path|None]

Returns a 4-tuple of saved paths:
(cleaned_rules_path, duplicates_report_path, spans_report_path, preview_pairs_path_or_None)
"""

from __future__ import annotations
import json, re, string, hashlib, math, os
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
from dataclasses import dataclass

# ---------- Optional YAML support ----------
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # YAML unavailable → will write JSON

# ---------- Text normalization helpers ----------
_PUNCT = str.maketrans("", "", string.punctuation)

def _norm_txt(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.translate(_PUNCT)
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_list_text(xs: Iterable[str] | None) -> tuple:
    if not xs:
        return tuple()
    return tuple(_norm_txt(x) for x in xs)

def _sig_scope(sc: dict) -> tuple:
    sc = sc or {}
    def _norm_list(xs): return tuple(sorted((_norm_txt(x) for x in (xs or []))))
    return (
        _norm_list(sc.get("actor")),
        _norm_list(sc.get("data_domain")),
        _norm_list(sc.get("context")),
    )

# ---------- Structural signature & de-dup ----------
def rule_signature(rule: dict, strict: bool = False) -> str:
    """
    Canonical signature for near-duplicate detection (structural).
    Default: ignores citation/span_id (so repeated obligations in one doc collapse).
    """
    src = rule.get("source", {}) or {}
    sc  = rule.get("scope", {}) or {}
    reqs = _norm_list_text(rule.get("requirements"))
    haz  = _norm_txt(rule.get("hazard"))
    exc  = tuple(sorted(_norm_list_text(rule.get("exceptions"))))
    cond = tuple(sorted(_norm_list_text(rule.get("conditions"))))
    sev  = (rule.get("severity") or "").lower().strip()
    scope_sig = _sig_scope(sc)

    base = (
        reqs, haz, exc, cond, sev, scope_sig,
        _norm_txt(src.get("doc") or "")
    )
    if strict:
        base = base + (_norm_txt(src.get("citation") or ""), _norm_txt(src.get("span_id") or ""))

    blob = json.dumps(base, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

def load_rules(path: Path) -> List[dict]:
    """
    Load rules from .yaml/.yml, .json (list or {"rules": [...]})
    OR .jsonl where each line is:
      - a bare rule dict, or
      - {"rule": {...}}, or
      - {"rules": [ ... ]}   (each appended)
    """
    if path.suffix.lower() == ".jsonl":
        rules: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "rule" in obj and isinstance(obj["rule"], dict):
                    rules.append(obj["rule"])
                elif isinstance(obj, dict) and "rules" in obj and isinstance(obj["rules"], list):
                    rules.extend(obj["rules"])
                elif isinstance(obj, dict):
                    rules.append(obj)
                else:
                    raise RuntimeError(f"{path} has a non-dict JSONL line; cannot interpret as a rule.")
        return rules

    # YAML / JSON files
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() in {".yml", ".yaml"} and yaml is not None:
            data = yaml.safe_load(text)  # type: ignore
        else:
            data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse {path}: {e}")

    if isinstance(data, dict) and "rules" in data:
        data = data["rules"]
    if not isinstance(data, list):
        raise RuntimeError(f"{path} does not contain a list of rules.")
    return data


def save_yaml_or_json(rules: List[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            out_json = out_path.with_suffix(".json")
            out_json.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
            return out_json
        out_path.write_text(
            yaml.safe_dump(rules, sort_keys=False, allow_unicode=True),  # type: ignore
            encoding="utf-8"
        )
        return out_path
    else:
        out_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

def structural_dedup(rules: List[dict], strict: bool = False) -> Tuple[List[dict], Dict[str, str], Dict[str, List[str]]]:
    """
    Returns (kept_rules, dup_map, kept_to_spans)
      - dup_map: duplicate_rule_id -> kept_rule_id
      - kept_to_spans: kept_rule_id -> list of all contributing span_ids
    """
    kept: List[dict] = []
    index: Dict[str, str] = {}           # sig -> kept_rule_id
    dup_map: Dict[str, str] = {}
    kept_to_spans: Dict[str, List[str]] = {}

    for r in rules:
        rid = r.get("rule_id") or ""
        sig = rule_signature(r, strict=strict)
        src = r.get("source", {}) or {}
        span = src.get("span_id") or ""

        if sig in index:
            dup_map[rid] = index[sig]
            kept_to_spans.setdefault(index[sig], []).append(span)
            continue

        index[sig] = rid
        kept.append(r)
        kept_to_spans.setdefault(rid, [])
        if span:
            kept_to_spans[rid].append(span)

    # de-duplicate span lists (keep order)
    for rid, spans in kept_to_spans.items():
        seen = set(); ordered = []
        for s in spans:
            if s and s not in seen:
                seen.add(s); ordered.append(s)
        kept_to_spans[rid] = ordered

    return kept, dup_map, kept_to_spans

# ---------- Semantic de-dup via embeddings ----------
def _get_openai_client():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set for semantic de-dup.")
    from openai import OpenAI  # type: ignore
    return OpenAI(api_key=key)

EMBED_MODEL = "text-embedding-3-large"

def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(x*x for x in b))
    return 0.0 if da == 0 or db == 0 else num/(da*db)

def _scope_block_key(rule: dict, doc_as_block: bool=True) -> Tuple:
    """Block by (doc, sorted actors, domains, contexts) to keep comparisons sensible."""
    src = (rule.get("source") or {})
    sc  = (rule.get("scope") or {})
    doc = (src.get("doc") or "").lower() if doc_as_block else ""
    return (
        doc,
        tuple(sorted((sc.get("actor") or []))),
        tuple(sorted((sc.get("data_domain") or []))),
        tuple(sorted((sc.get("context") or []))),
    )

def _semantic_string(rule: dict) -> str:
    """Compact canonical text for embedding."""
    req = " | ".join(rule.get("requirements", []) or [])
    haz = rule.get("hazard","")
    cond= " | ".join(rule.get("conditions", []) or [])
    exc = " | ".join(rule.get("exceptions", []) or [])
    sev = rule.get("severity","")
    sc  = rule.get("scope", {}) or {}
    scope = f"ACT:{','.join(sc.get('actor',[]))};DOM:{','.join(sc.get('data_domain',[]))};CTX:{','.join(sc.get('context',[]))}"
    return f"REQ:{req} HAZ:{haz} COND:{cond} EXC:{exc} SEV:{sev} {scope}"

def _embed_batch(client, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]  # type: ignore[attr-defined]

@dataclass
class PairSim:
    a_idx: int
    b_idx: int
    sim: float

def semantic_dedup(
    rules: list[dict],
    kept_to_spans: dict[str, list[str]] | None = None,
    threshold: float = 0.9,
    doc_block: bool = True,
    preview: bool = False,
    preview_k: int = 200,
    preview_path: Path | None = None,
) -> tuple[list[dict], dict[str, str], Path | None]:
    """
    Embedding-based near-duplicate merge (semantic).
    - Blocks by (doc,scope) to avoid spurious merges
    - Greedy keep-first clustering with cosine ≥ threshold
    - Optionally writes a preview TSV of top similar pairs across all blocks
    Returns: (kept_rules, dup_map, preview_path_or_None)
    """
    if not rules:
        return rules, {}, None

    client = _get_openai_client()
    dup_map: dict[str, str] = {}
    kept_mask = [True]*len(rules)
    preview_pairs: list[PairSim] = []

    # 1) Build blocks
    blocks: dict[Tuple, list[int]] = {}
    for i, r in enumerate(rules):
        blocks.setdefault(_scope_block_key(r, doc_as_block=doc_block), []).append(i)

    # 2) Process each block
    for key, idxs in blocks.items():
        if len(idxs) < 2:
            continue

        texts = [_semantic_string(rules[i]) for i in idxs]
        embs  = _embed_batch(client, texts)

        taken = set()
        for a_pos, a_idx in enumerate(idxs):
            if a_idx in taken:
                continue
            a_emb = embs[a_pos]
            a_rid = rules[a_idx].get("rule_id") or f"rule_{a_idx}"
            for b_pos, b_idx in enumerate(idxs[a_pos+1:], start=a_pos+1):
                if b_idx in taken:
                    continue
                sim = _cosine(a_emb, embs[b_pos])

                # collect for preview
                if preview:
                    preview_pairs.append(PairSim(a_idx, b_idx, sim))

                if sim >= threshold:
                    taken.add(b_idx)
                    kept_mask[b_idx] = False
                    b_rid = rules[b_idx].get("rule_id") or f"rule_{b_idx}"
                    dup_map[b_rid] = a_rid
                    if kept_to_spans is not None:
                        src = rules[b_idx].get("source", {}) or {}
                        span = src.get("span_id") or ""
                        if span:
                            kept_to_spans.setdefault(a_rid, [])
                            if span not in kept_to_spans[a_rid]:
                                kept_to_spans[a_rid].append(span)

    kept_rules = [r for r, keep in zip(rules, kept_mask) if keep]

    # 3) Optional preview file (top-K pairs)
    final_preview_path: Path | None = None
    if preview and preview_pairs:
        preview_pairs.sort(key=lambda p: p.sim, reverse=True)
        rows = preview_pairs[:preview_k]
        final_preview_path = preview_path or Path("semantic_pairs_preview.tsv")
        with final_preview_path.open("w", encoding="utf-8") as f:
            f.write("sim\ta_rule_id\ta_req\tb_rule_id\tb_req\n")
            def _req(r): return " | ".join(r.get("requirements", []) or [])
            for p in rows:
                ar = rules[p.a_idx]; br = rules[p.b_idx]
                f.write(f"{p.sim:.4f}\t{ar.get('rule_id','')}\t{_req(ar)}\t{br.get('rule_id','')}\t{_req(br)}\n")
    return kept_rules, dup_map, final_preview_path

# ---------- Orchestration ----------
def default_output_path(inputs: List[Path]) -> Path:
    """
    If output_path not provided, write next to the FIRST input with suffix '.cleaned.yaml'
    (or '.cleaned.json' if PyYAML is unavailable).
    """
    base = inputs[0]
    suffix = ".cleaned.yaml" if yaml is not None else ".cleaned.json"
    return base.with_suffix(suffix)

def run_dedup(
    inputs: List[Path],
    output_path: Path | None = None,
    *,
    strict: bool = False,
    semantic: bool = True,
    threshold: float = 0.90,
    doc_block: bool = True,
    preview: bool = False,
    preview_k: int = 200,
    preview_path: Path | None = None,
) -> tuple[Path, Path, Path, Path | None]:
    """
    Run structural (+ optional semantic) de-dup and write:
      - cleaned rules (YAML/JSON based on output extension and PyYAML availability)
      - <output>.dups.tsv       (duplicate_rule_id -> kept_rule_id)
      - <output>.spans.tsv      (kept_rule_id -> all contributing span_ids)
      - (optional) <output>.pairs.tsv (top similar pairs) if preview=True

    Returns a 4-tuple:
      (cleaned_rules_path, duplicates_report_path, spans_report_path, preview_pairs_path_or_None)
    """
    if not inputs:
        raise ValueError("inputs must contain at least one path")
    inputs = [Path(p) for p in inputs]

    # 1) Load
    all_rules: List[dict] = []
    for p in inputs:
        all_rules.extend(load_rules(p))

    # 2) Structural de-dup
    kept, dup_map_struct, kept_spans = structural_dedup(all_rules, strict=strict)

    # 3) Semantic de-dup (optional)
    final_rules = kept
    dup_map_all = dict(dup_map_struct)
    preview_file: Path | None = None
    if semantic:
        try:
            final_rules, dup_map_sem, sem_preview = semantic_dedup(
                kept,
                kept_to_spans=kept_spans,
                threshold=threshold,
                doc_block=doc_block,
                preview=preview,
                preview_k=preview_k,
                preview_path=preview_path,
            )
            preview_file = sem_preview
            dup_map_all.update(dup_map_sem)
        except Exception as e:
            # Semantic step is optional; continue with structural results.
            preview_file = None

    # 4) Write cleaned rules
    out_rules_path = output_path or default_output_path(inputs)
    out_rules_path = save_yaml_or_json(final_rules, out_rules_path)

    # 5) Write duplicates report
    dups_path = out_rules_path.with_suffix(out_rules_path.suffix + ".dups.tsv")
    with dups_path.open("w", encoding="utf-8") as f:
        f.write("duplicate_rule_id\tkept_rule_id\n")
        for d, k in dup_map_all.items():
            f.write(f"{d}\t{k}\n")

    # 6) Write span provenance
    spans_path = out_rules_path.with_suffix(out_rules_path.suffix + ".spans.tsv")
    with spans_path.open("w", encoding="utf-8") as f:
        f.write("kept_rule_id\tspan_id\n")
        for rid, spans in kept_spans.items():
            for s in spans:
                f.write(f"{rid}\t{s}\n")

    # 7) If preview requested but no explicit path and none created yet, place next to output
    if preview and preview_file is None:
        preview_file = out_rules_path.with_suffix(out_rules_path.suffix + ".pairs.tsv")
        # Nothing to write if there were no pairs; ensure empty file exists to signal "preview was requested"
        preview_file.write_text("sim\ta_rule_id\ta_req\tb_rule_id\tb_req\n", encoding="utf-8")

    return out_rules_path, dups_path, spans_path, preview_file