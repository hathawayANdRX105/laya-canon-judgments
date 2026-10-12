#!/usr/bin/env python3
"""Group-aware train/val/test split for laya judgment rows.

Splits by GROUP KEY (state hash / doc id), never by row: every row that
belongs to one state lands in the same split, so near-duplicates cannot
leak across train/validation/test.

Usage:
    make_splits.py rows.jsonl --seed 42 --ratios 0.8,0.1,0.1 --out-dir data/splits
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def group_key(row: dict, key: str) -> str:
    if key != "auto" and row.get(key):
        return str(row[key])
    # deterministic fallback: hash the state payload
    return hashlib.sha256(
        json.dumps(row.get("state", {}), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows", help="rows JSONL (one judgment state per line)")
    ap.add_argument("--group-key", default="state_hash",
                    help="field to group by (default: state_hash; 'auto' hashes state)")
    ap.add_argument("--ratios", default="0.8,0.1,0.1",
                    help="train,validation,test ratios (must sum to 1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/splits")
    args = ap.parse_args()

    train_r, val_r, test_r = (float(x) for x in args.ratios.split(","))
    assert abs(train_r + val_r + test_r - 1.0) < 1e-9, "ratios must sum to 1"

    rows = []
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(group_key(row, args.group_key), []).append(row)

    keys = sorted(groups)  # deterministic order before shuffle
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = round(n * train_r)
    n_val = round(n * val_r)
    splits = {
        "train": keys[:n_train],
        "validation": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups_per_split = {}
    rows_per_split = {}
    for name, ks in splits.items():
        out_rows = [r for k in ks for r in groups[k]]
        groups_per_split[name] = len(ks)
        rows_per_split[name] = len(out_rows)
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "seed": args.seed,
        "ratios": [train_r, val_r, test_r],
        "group_key": args.group_key,
        "groups_per_split": groups_per_split,
        "rows_per_split": rows_per_split,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
