#!/usr/bin/env python3
"""Extract per-commit diff states from a git repo.

Each commit becomes one judgment state — the exact material the
canon gate judges. lang follows the gate's own routing rule: Han
characters anywhere -> zh (multilingual checkpoint), else en
(typed-decisions checkpoint).

Usage:
    gen_states.py /path/to/repo --out states.jsonl [--limit N] [--max-chars 30000]
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HAN = re.compile(r"[一-鿿]")


def commit_list(repo: Path, limit: int) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--first-parent", f"HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return out[:limit] if limit > 0 else out


def commit_diff(repo: Path, commit: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "show", "--format=", "--no-ext-diff", commit],
        capture_output=True, text=True, check=True,
    ).stdout
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max states (0 = all)")
    ap.add_argument("--max-chars", type=int, default=30000,
                    help="truncate longer diffs (router caps tokens anyway)")
    args = ap.parse_args()

    repo = Path(args.repo)
    rows = []
    for commit in commit_list(repo, args.limit):
        diff = commit_diff(repo, commit)
        if not diff.strip():
            continue
        if len(diff) > args.max_chars:
            diff = diff[:args.max_chars]
        lang = "zh" if HAN.search(diff) else "en"
        state_hash = hashlib.sha256(diff.encode()).hexdigest()[:16]
        rows.append({
            "state_hash": state_hash,
            "lang": lang,
            "source": f"{repo.name}:{commit[:12]}",
            "state": diff,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    langs = {}
    for r in rows:
        langs[r["lang"]] = langs.get(r["lang"], 0) + 1
    print(json.dumps({"states": len(rows), "by_lang": langs}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
