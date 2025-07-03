# robust_pdf_ingest.py
# End-to-end hardened PDF → JSONL (sentences, tables, captions, figure summaries)
# Safe on Windows; OpenAI optional; images/tables can be disabled if needed.

from __future__ import annotations

import os, re, json, math, hashlib, argparse, statistics, base64, sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# --- Third-party deps ---
# pip install pymupdf pdfplumber ftfy nltk
# (optional) pip install openai
import fitz  # PyMuPDF
from ftfy import fix_text

# pdfplumber is optional (for tables) – we gate its import
_HAS_PDFPLUMBER = True
try:
    import pdfplumber
except Exception:
    _HAS_PDFPLUMBER = False

# -------- Sentence splitter (NLTK optional) --------
_USE_NLTK = False
try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    _USE_NLTK = True
except Exception:
    _USE_NLTK = False

# ====================== Small helpers ======================

def sent_split(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if _USE_NLTK:
        try:
            return [s.strip() for s in _nltk_sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    # Regex fallback: avoids splitting on initials/ALLCAPS abbrevs; not perfect but robust.
    return [s.strip() for s in re.split(r"(?<!\b[A-Z])(?<=[.!?])\s+(?=[A-Z(])", text) if s.strip()]

def chunked(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]

def sha1_8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8]

def slug(s: str, max_len: int = 60) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len] or "section"

def dehyphen(text: str) -> str:
    # join hyphenated linebreaks and clean multiple newlines
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text

def normalize_spaces(text: str) -> str:
    # collapse weird spacing; ftfy handles ligatures & encoding weirdness
    text = fix_text(text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


# ====================== Sectioning (TOC / headings) ======================

def build_section_map_from_toc(doc: fitz.Document) -> Dict[int, List[str]]:
    """
    Returns map: start_page_index (0-based) -> section_path (list of str).
    """
    toc = doc.get_toc(simple=False) or []  # [level, title, page, ...]
    sec_map: Dict[int, List[str]] = {}
    stack: List[str] = []
    for entry in toc:
        level, title, page_num = entry[0], entry[1], entry[2]
        if not isinstance(page_num, int):
            continue
        while len(stack) >= level:
            stack.pop()
        stack.append((title or "").strip())
        sec_map[page_num - 1] = stack.copy()
    return sec_map

def infer_headings_when_no_toc(doc: fitz.Document) -> Dict[int, List[str]]:
    """
    Heuristic: use large-font spans on a page as headings; carry forward until next heading.
    Returns start_page_index -> section_path (shallow).
    """
    sec_map: Dict[int, List[str]] = {}
    current_path: List[str] = []
    for i, page in enumerate(doc):
        try:
            d = page.get_text("dict")
        except Exception:
            sec_map[i] = current_path.copy() if current_path else [f"Page {i+1}"]
            continue

        sizes = []
        for b in d.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    sz = s.get("size", 0)
                    if sz: sizes.append(sz)

        if not sizes:
            sec_map[i] = current_path.copy() if current_path else [f"Page {i+1}"]
            continue

        top = sorted(sizes, reverse=True)[:6]
        thr = statistics.median(top) if top else 0
        candidates = []
        for b in d.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    if s.get("size", 0) >= thr:
                        txt = (s.get("text", "") or "").strip()
                        # Avoid pages where body text is also large—apply length heuristics
                        if 4 <= len(txt) <= 120 and len(txt.split()) <= 18:
                            candidates.append(txt)
        if candidates:
            current_path = [candidates[0]]
        sec_map[i] = current_path.copy() if current_path else [f"Page {i+1}"]
    return sec_map


# ====================== Header / Footer ======================

def learn_repeating_lines(doc: fitz.Document, sample_pages: int = 10) -> set[str]:
    """
    Sample first N pages, hash full lines; any repeating line appearing on >= 1/2 sampled pages
    is considered header/footer (to be filtered).
    """
    sample_pages = min(max(sample_pages, 1), len(doc))
    counts = Counter()
    for i in range(sample_pages):
        try:
            raw = doc[i].get_text("text", flags=fitz.TEXT_INHIBIT_SPACES)
        except Exception:
            raw = doc[i].get_text("text")
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln: 
                continue
            counts[sha1_8(ln)] += 1
    cutoff = max(2, sample_pages // 2)
    return {h for h, c in counts.items() if c >= cutoff}

def filter_repeating_lines(text: str, common_hashes: set[str]) -> str:
    kept = []
    for ln in text.splitlines():
        if sha1_8(ln.strip()) in common_hashes:
            continue
        kept.append(ln)
    return "\n".join(kept)


# ====================== Columns & Blocks ======================

def cluster_columns_by_x(blocks: List[Tuple[float,float,float,float,str]]) -> List[List[Tuple]]:
    """
    Fallback: cluster block rectangles by their x0 using a 2-cluster heuristic.
    Returns list of columns (each is list of blocks), left-to-right order.
    """
    if len(blocks) <= 1:
        return [blocks]
    blocks = sorted(blocks, key=lambda b: (round(b[0],1), round(b[1],1)))
    xs = [b[0] for b in blocks]
    mid = statistics.median(xs)
    left = [b for b in blocks if b[0] <= mid]
    right = [b for b in blocks if b[0] > mid]
    if not left or not right:
        return [sorted(blocks, key=lambda b:(b[1], b[0]))]
    return [
        sorted(left, key=lambda b:(b[1], b[0])),
        sorted(right, key=lambda b:(b[1], b[0]))
    ]

def extract_page_text_by_columns(page: fitz.Page, footer_margin: int = 50, top_margin: int = 30) -> str:
    """
    General approach: use text blocks, filter margins, cluster by x (columns), then read each column top→bottom.
    """
    text = ""
    try:
        blks = page.get_text("blocks", sort=True, flags=fitz.TEXT_INHIBIT_SPACES)
    except Exception:
        blks = page.get_text("blocks", sort=True)

    page_h = page.rect.height
    usable = [b for b in blks if (b[1] > top_margin and b[3] < page_h - footer_margin and str(b[4]).strip())]
    if not usable:
        usable = [b for b in blks if str(b[4]).strip()]
    cols = cluster_columns_by_x(usable)
    for col in cols:
        for x0,y0,x1,y1,txt, *_ in col:
            text += str(txt).rstrip() + "\n"
    return text


# ====================== Captions & Tables ======================

CAPTION_PAT = re.compile(r"^\s*(Figure|Fig\.|Table)\s+(\d+|[A-Za-z]\d*|\d+\.\d+)\b", re.I)

@dataclass
class Caption:
    kind: str
    label: str
    text: str
    bbox: Tuple[float,float,float,float]

def find_captions(page: fitz.Page) -> List[Caption]:
    caps = []
    try:
        blks = page.get_text("blocks", sort=True, flags=fitz.TEXT_INHIBIT_SPACES)
    except Exception:
        blks = page.get_text("blocks", sort=True)
    for x0,y0,x1,y1,txt,*_ in blks:
        t = str(txt).strip()
        if not t: 
            continue
        m = CAPTION_PAT.match(t)
        if m:
            caps.append(Caption(kind=m.group(1).lower(), label=m.group(2), text=t, bbox=(x0,y0,x1,y1)))
    return caps

def extract_tables_with_pdfplumber(path: str) -> Dict[int, List[Dict]]:
    """
    Use pdfplumber to extract tables with rows.
    Returns map page_number(1-based) -> list of table dicts.
    """
    if not _HAS_PDFPLUMBER:
        return {}
    out = defaultdict(list)
    with pdfplumber.open(path) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables() or []
                for ti, t in enumerate(tables):
                    rows = [[(cell or "").strip() for cell in row] for row in t]
                    out[pi].append({"table_index": ti, "rows": rows})
            except Exception:
                continue
    return out


# ====================== OpenAI Vision (optional) ======================

def summarize_figure_with_openai(image_bytes: bytes,
                                 caption_snippet: str,
                                 api_key: Optional[str],
                                 model: str = "gpt-5") -> Optional[str]:
    """
    Uses GPT-5 to produce a 1–2 sentence summary of an image, with optional caption context.
    Requires OpenAI Python SDK v1.x.
    """
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        user_parts = [
            {"type": "text",
             "text": ("You are summarizing figures from a policy/standards PDF. "
                      "Provide a concise 1–2 sentence description relevant to compliance or governance. "
                      "Avoid speculation; rely on visible content and caption context.")},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
        if caption_snippet:
            user_parts.append({"type": "text",
                               "text": f"Caption context: {caption_snippet[:500]}"})

        resp = client.chat.completions.create(
            model=model,  # <-- GPT-5 (latest)
            messages=[{"role": "user", "content": user_parts}],
            temperature=0.0,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        logger.exception("OpenAI vision summarization failed: %s", e)
        return None


def extract_image_png_bytes(page: fitz.Page, xref: int) -> Optional[bytes]:
    """
    Render image referenced by xref to PNG bytes (lossless).
    Strongly guarded to avoid native crashes on Windows.
    """
    try:
        pix = fitz.Pixmap(page.parent, xref)
        if pix is None:
            return None
    except Exception:
        # try bbox-based rasterization
        try:
            rects = page.get_image_bbox(xref) or []
            if not rects:
                return None
            rect = rects if isinstance(rects, fitz.Rect) else rects[0]
            mat = fitz.Matrix(150/72, 150/72)  # ~150 dpi
            pix = page.get_pixmap(matrix=mat, clip=rect)
        except Exception:
            return None
    try:
        if (getattr(pix, "alpha", 0)) or (getattr(pix, "colorspace", None) and pix.colorspace.n > 3):
            pix = fitz.Pixmap(fitz.csRGB, pix)
        return pix.tobytes("png")
    except Exception:
        return None


# ====================== Main extraction ======================

import os, json, time, logging
from typing import Optional, Dict, List
from collections import defaultdict
import fitz  # PyMuPDF

# assume these helpers already exist in your module:
# build_section_map_from_toc, infer_headings_when_no_toc, learn_repeating_lines,
# extract_tables_with_pdfplumber, _HAS_PDFPLUMBER,
# extract_page_text_by_columns, filter_repeating_lines, normalize_spaces, dehyphen,
# find_captions, Caption, summarize_figure_with_openai, extract_image_png_bytes,
# sent_split, slug

# module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

def _timenow() -> float:
    return time.perf_counter()

def robust_ingest_pdf(pdf_path: str,
                      out_jsonl: Optional[str] = None,
                      doc_id: Optional[str] = None,
                      openai_api_key: Optional[str] = None,
                      openai_model: str = "gpt-4o-mini",
                      learn_header_footer_pages: int = 10,
                      footnote_bottom_pct: float = 0.10,
                      small_font_threshold_pct: float = 0.20,
                      enable_images: bool = True,
                      enable_tables: bool = True,
                      out_base_dir: str = "out",
                      group_n: int = 4) -> str:
    """
    Stream JSONL with records:
    {span_id, kind, section, section_path, page, text, meta}
    kind ∈ {"sentence","table","caption","figure_summary","error"}

    Flags:
      - enable_images: set False to skip image/caption processing entirely (safest on Windows).
      - enable_tables: set False to skip pdfplumber table extraction.
    """
    start_t = _timenow()
    logger.info("Starting robust_ingest_pdf(pdf_path=%r, out_jsonl=%r, images=%s, tables=%s)",
                pdf_path, out_jsonl, enable_images, enable_tables)

    if not os.path.exists(pdf_path):
        logger.error("PDF not found: %s", pdf_path)
        raise FileNotFoundError(pdf_path)

    doc = fitz.open(pdf_path)
    raw_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_id = doc_id or raw_name
    cleaned_name = slug(raw_name) or "document"

    # Decide output path: out/<pdf_cleaned_name>/spans.jsonl (and ensure dirs exist)
    if not out_jsonl:
        out_dir = os.path.join(out_base_dir, cleaned_name)
        os.makedirs(out_dir, exist_ok=True)
        out_jsonl = os.path.join(out_dir, "spans.jsonl")
    else:
        if os.path.isdir(out_jsonl):
            out_jsonl = os.path.join(out_jsonl, "spans.jsonl")
        os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    logger.info("Opened PDF id=%s pages=%d -> writing to %s", doc_id, doc.page_count, out_jsonl)


    # Section map
    sec_map = build_section_map_from_toc(doc)
    if not sec_map:
        logger.debug("No TOC-based section map; inferring headings.")
        sec_map = infer_headings_when_no_toc(doc)
    logger.info("Section map learned with %d section starts.", len(sec_map))

    # Header/footer hashes
    hf_t0 = _timenow()
    common_hashes = learn_repeating_lines(doc, sample_pages=learn_header_footer_pages)
    logger.info("Learned repeating header/footer lines: %d (%.3fs)",
                len(common_hashes), _timenow() - hf_t0)

    # Pre-extract tables via pdfplumber (optional)
    tables_by_page: Dict[int, List[Dict]] = {}
    if enable_tables and '_HAS_PDFPLUMBER' in globals() and _HAS_PDFPLUMBER:
        tb_t0 = _timenow()
        try:
            tables_by_page = extract_tables_with_pdfplumber(pdf_path) or {}
            logger.info("Extracted tables via pdfplumber for %d pages (%.3fs).",
                        len(tables_by_page), _timenow() - tb_t0)
        except Exception:
            logger.exception("pdfplumber table extraction failed; continuing without tables.")
            tables_by_page = {}
    elif enable_tables:
        logger.warning("enable_tables=True but pdfplumber is unavailable; skipping tables.")

    total_counts = defaultdict(int)  # sentence, table, caption, figure_summary, error
    pages_processed = 0

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        # Per-section sentence counters
        sent_counters: Dict[str, int] = defaultdict(int)

        for page in doc:
            page_i = page.number   # 0-based
            page_no = page_i + 1

            # Determine current section path (last start ≤ this page)
            starts = sorted(sec_map.keys())
            sec_path = sec_map[starts[0]] if starts else [f"Page {page_no}"]
            for sp in starts:
                if page_i >= sp:
                    sec_path = sec_map[sp]

            try:
                # slug can crash: compute inside try and guard
                try:
                    sec_path_slug = "/".join(slug(s) for s in sec_path)
                except Exception:
                    logger.exception("Page %d: slug() failed; using fallback.", page_no)
                    sec_path_slug = f"page_{page_no:04d}"

                logger.info("Page %d/%d: section=%r", page_no, doc.page_count, " / ".join(sec_path))

                # -------- 1) TEXT by columns (defensive)
                try:
                    raw_text = extract_page_text_by_columns(page)
                except Exception:
                    logger.exception("Page %d: extract_page_text_by_columns failed; falling back to page.get_text('text').", page_no)
                    raw_text = None

                if not isinstance(raw_text, str):
                    try:
                        raw_text = page.get_text("text") or ""
                    except Exception:
                        logger.exception("Page %d: page.get_text('text') failed; continuing with empty text.", page_no)
                        raw_text = ""

                # normalize pipeline defensively
                try:
                    raw_text = filter_repeating_lines(raw_text, common_hashes)
                except Exception:
                    logger.exception("Page %d: filter_repeating_lines failed; keeping original text.", page_no)

                try:
                    raw_text = dehyphen(raw_text if isinstance(raw_text, str) else str(raw_text))
                except Exception:
                    logger.exception("Page %d: dehyphen failed; keeping un-dehyphenated text.", page_no)

                try:
                    raw_text = normalize_spaces(raw_text if isinstance(raw_text, str) else str(raw_text))
                except Exception:
                    logger.exception("Page %d: normalize_spaces failed; leaving text as-is.", page_no)

                # -------- 2) Footnote filtering (guard every field)
                try:
                    d = page.get_text("dict") or {"blocks": []}
                except Exception:
                    logger.debug("Page %d: get_text('dict') failed; skipping footnote filtering.", page_no)
                    d = {"blocks": []}

                try:
                    page_h = float(getattr(page.rect, "height", 0.0) or 0.0)
                except Exception:
                    page_h = 0.0

                try:
                    small_sizes = []
                    for b in d.get("blocks", []) or []:
                        for l in b.get("lines", []) or []:
                            for s in l.get("spans", []) or []:
                                sz = s.get("size", 0) or 0
                                if sz: small_sizes.append(sz)
                    thr = 0.0
                    if small_sizes:
                        sorted_sizes = sorted(s for s in small_sizes if s > 0)
                        if sorted_sizes:
                            idx = max(0, int(len(sorted_sizes) * small_font_threshold_pct) - 1)
                            thr = float(sorted_sizes[idx])

                    kept_lines = []
                    for b in d.get("blocks", []) or []:
                        for l in b.get("lines", []) or []:
                            spans = l.get("spans", []) or []
                            if not spans:
                                continue
                            try:
                                bbox = [(s.get("bbox") or [0,0,0,0]) for s in spans]
                                y_bot = max(bb[3] for bb in bbox) if bbox else 0
                            except Exception:
                                y_bot = 0
                            use_line = True
                            if page_h and y_bot >= page_h * (1.0 - footnote_bottom_pct):
                                try:
                                    if all(((s.get("size") or 0) <= thr) for s in spans):
                                        use_line = False
                                except Exception:
                                    pass
                            if use_line:
                                kept_lines.append("".join((s.get("text") or "") for s in spans))
                    if kept_lines:
                        try:
                            raw_text = normalize_spaces(dehyphen("\n".join(kept_lines)))
                        except Exception:
                            raw_text = "\n".join(kept_lines)
                except Exception:
                    logger.exception("Page %d: footnote filtering failed; continuing.", page_no)

                # -------- 3) CAPTIONS & IMAGES (guarded) — will be skipped if enable_images=False
                captions: List['Caption'] = []
                if enable_images:
                    try:
                        captions = find_captions(page) or []
                        logger.debug("Page %d: found %d captions.", page_no, len(captions))
                    except Exception:
                        logger.exception("Page %d: caption detection failed.", page_no)
                        captions = []

                    for ci, cap in enumerate(captions):
                        try:
                            sid = f"{doc_id}/{sec_path_slug}/caption_{page_no}_{ci:02d}"
                            rec = {
                                "span_id": sid,
                                "kind": "caption",
                                "section": sec_path[-1] if sec_path else "",
                                "section_path": sec_path,
                                "page": page_no,
                                "text": cap.text,
                                "meta": {"kind": getattr(cap, 'kind', None), "label": getattr(cap, 'label', None), "bbox": getattr(cap, 'bbox', None)}
                            }
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            total_counts["caption"] += 1
                        except Exception:
                            logger.exception("Page %d: caption emission failed.", page_no)

                    if openai_api_key:
                        try:
                            images = page.get_images(full=True) or []
                        except Exception:
                            logger.exception("Page %d: get_images failed.", page_no)
                            images = []
                        for ii, imeta in enumerate(images):
                            try:
                                xref = imeta[0]
                                png_bytes = extract_image_png_bytes(page, xref)
                                if not png_bytes:
                                    continue
                                cap_text = captions[0].text if captions else ""
                                summary = summarize_figure_with_openai(png_bytes, cap_text, openai_api_key, openai_model) or ""
                                sid = f"{doc_id}/{sec_path_slug}/figure_{page_no}_{ii:02d}_summary"
                                rec = {
                                    "span_id": sid,
                                    "kind": "figure_summary",
                                    "section": sec_path[-1] if sec_path else "",
                                    "section_path": sec_path,
                                    "page": page_no,
                                    "text": summary if summary else "(no summary)",
                                    "meta": {"has_image": True}
                                }
                                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                total_counts["figure_summary"] += 1
                            except Exception:
                                logger.exception("Page %d: image %d summarization failed.", page_no, ii)
                                continue

                # -------- 4) SENTENCES (grouped)
                groups_on_page = 0
                try:
                    text_for_split = raw_text if isinstance(raw_text, str) else str(raw_text or "")
                    sents = [s for s in (sent_split(text_for_split) or []) if s]
                    for group in chunked(sents, max(1, int(group_n))):
                        key = "/".join(sec_path)
                        # advance a section-local counter ONCE per group
                        sent_counters[key] += 1
                        # NOTE: keep the identifier prefix as 'sent_' for compatibility, or rename to 'sgrp_' if you prefer
                        sid = f"{doc_id}/{sec_path_slug}/sent_{sent_counters[key]:04d}"
                        rec = {
                            "span_id": sid,
                            "kind": "sentence",  # or "sentence_group" if you want to distinguish
                            "section": sec_path[-1] if sec_path else "",
                            "section_path": sec_path,
                            "page": page_no,
                            "text": " ".join(group),
                            "meta": {
                                "group_size": len(group),
                                "group_n": int(group_n)
                            }
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        groups_on_page += 1
                        # Count grouped spans, not raw sentences, in totals
                        total_counts["sentence"] += 1
                except Exception:
                    logger.exception("Page %d: grouped sentence emission failed.", page_no)


                # -------- 5) TABLES
                tables_on_page = 0
                try:
                    if enable_tables and tables_by_page:
                        page_tables = tables_by_page.get(page_no, []) or []
                        for ti, t in enumerate(page_tables):
                            rows = t.get("rows") or []
                            if not rows:
                                continue
                            flat = "\n".join([", ".join((r or [])) for r in rows])
                            sid = f"{doc_id}/{sec_path_slug}/table_{page_no}_{ti:02d}"
                            rec = {
                                "span_id": sid,
                                "kind": "table",
                                "section": sec_path[-1] if sec_path else "",
                                "section_path": sec_path,
                                "page": page_no,
                                "text": flat,
                                "meta": {"rows": rows}
                            }
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            tables_on_page += 1
                            total_counts["table"] += 1
                except Exception:
                    logger.exception("Page %d: table emission failed.", page_no)

                pages_processed += 1
                logger.info("Page %d processed | sentences=%d captions=%d tables=%d",
                            page_no, groups_on_page, len(captions), tables_on_page)

            except MemoryError:
                logger.exception("Page %d: MemoryError; aborting ingest.", page_no)
                raise
            except Exception as e:
                # ensure we can always write an error record even if slug failed
                sec_slug = locals().get("sec_path_slug", f"page_{page_no:04d}")
                sid = f"{doc_id}/{sec_slug}/page_{page_no:04d}_error"
                rec = {
                    "span_id": sid,
                    "kind": "error",
                    "section": sec_path[-1] if sec_path else "",
                    "section_path": sec_path,
                    "page": page_no,
                    "text": f"Extraction error: {type(e).__name__}: {e}"
                }
                try:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                total_counts["error"] += 1
                logger.exception("Page %d: Unhandled extraction error; wrote error record.", page_no)

    dur = _timenow() - start_t
    logger.info(
        "Completed ingest for %s in %.3fs | pages=%d sentences=%d captions=%d figure_summaries=%d tables=%d errors=%d -> %s",
        doc_id, dur, pages_processed,
        total_counts["sentence"], total_counts["caption"],
        total_counts["figure_summary"], total_counts["table"], total_counts["error"],
        out_jsonl
    ) 
    return out_jsonl