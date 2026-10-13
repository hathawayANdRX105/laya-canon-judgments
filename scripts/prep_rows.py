#!/usr/bin/env python3
"""Merge judgment row sources and fill the eval `expected` label.

The evaluator (laya/evals.py) scores against a per-question `expected`
value (choice -> option string, noul -> bool). Seed rows carry it; the
screened rows do not, so it is derived as the argmax of the gold
distribution (the teacher's most likely answer).

Usage:
    prep_rows.py rows.jsonl kept.jsonl --out merged.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def expected_from_gold(gold: dict) -> dict:
    out = {}
    for qid, dist in gold.items():
        probs = dist.get("probabilities", {})
        if not probs:
            continue
        if len(probs) == 2 and set(probs) == {"true", "false"}:
            out[qid] = probs["true"] >= probs["false"]
        else:
            out[qid] = max(sorted(probs), key=probs.get)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="+",
                    help="row JSONL files (later files lose duplicate "
                         "state_hash rows to earlier ones)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seen: set[str] = set()
    kept_rows, derived = 0, 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for src in args.sources:
            for line in Path(src).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                h = row.get("state_hash")
                if h and h in seen:
                    continue
                seen.add(h or "")
                if not row.get("expected"):
                    row["expected"] = expected_from_gold(row.get("gold", {}))
                    derived += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_rows += 1
    print(f"{kept_rows} rows -> {args.out} "
          f"({derived} derived expected)", file=sys.stderr)


if __name__ == "__main__":
    main()
