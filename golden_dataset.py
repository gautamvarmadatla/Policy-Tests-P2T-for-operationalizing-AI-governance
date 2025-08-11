# Create gold.csv from NIST.AI.100-1.jsonl (NO filtering, NO dedup, include ALL rows)
# Prereqs: pip install pandas

import json, os
from pathlib import Path
import pandas as pd

INPUT_JSONL = Path(os.path.expanduser(r"~/Downloads/NIST.AI.100-1.candidates.jsonl"))
OUT_CSV     = Path(r"D:\AAAI ( AIGOV 26 )\gold.csv")

rows = []
with INPUT_JSONL.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f, start=1):
        try:
            rec = json.loads(line)
        except Exception:
            # keep even unparsable lines as placeholders
            rec = {}

        # Pull whatever is present; do NOT drop anything
        span_id  = rec.get("span_id") or rec.get("id") or f"nist-{idx:06d}"
        src      = rec.get("source") or {}
        doc      = src.get("doc") or rec.get("doc") or ""
        citation = src.get("citation") or rec.get("citation") or ""
        text     = (rec.get("text") or "")

        rows.append({
            "clause_id": span_id,
            "doc": doc,
            "citation": citation,
            "text": text,

            # Blank columns for annotators to fill (do not prefill/remove anything)
            "is_testable": "",
            "scope.actor": "",
            "scope.data_domain": "",
            "scope.context": "",
            "hazard": "",
            "requirements": "",
            "exceptions": "",
            "severity": ""
        })

# Ensure folder exists and write CSV with exact headers
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
cols = ["clause_id","doc","citation","text",
        "is_testable","scope.actor","scope.data_domain","scope.context",
        "hazard","requirements","exceptions","severity"]
pd.DataFrame(rows, columns=cols).to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"✓ Wrote {len(rows)} rows to {OUT_CSV}")
