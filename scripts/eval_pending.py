#!/usr/bin/env python3
"""Evaluate a pending-states queue against a running laya-serve.

CI-side counterpart of screen.py stage 3: the pending queue is exported
locally (`screen.py --export-pending`) and evaluated here on a runner
after starting the server (scripts/serve_cpu.py).

Pending rows: {"state_hash", "source", "lang", "state"} one per line.
Output: cache fragment {"results": {...}, "errors": {...}} shaped like
data/screened/.eval-cache.json; merge locally with
`screen.py --merge-fragment <file>`.

Requests are sequential: the bundled server runs inference on a
single-worker pool, so concurrency only manufactures 503s.

Usage: eval_pending.py [--pending data/pending.jsonl]
                       [--out data/cache-fragment.json]
                       [--api http://127.0.0.1:8000]
"""

import argparse
import json
import time
from pathlib import Path

from jevclient import (ask_jev, confidence_of, gate_questions,
                       gold_from_answers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pending", default="data/pending.jsonl")
    ap.add_argument("--out", default="data/cache-fragment.json")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="laya-local")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    questions = {
        lang: gate_questions(json.loads(
            (root / f"data/questions_{lang}_done_when.json").read_text()))
        for lang in ("en", "zh")
    }
    rows = [json.loads(l) for l in
            (root / args.pending).read_text().splitlines() if l]
    print(f"pending: {len(rows)} states", flush=True)

    results, errors = {}, {}
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        try:
            ans = ask_jev(args.api, args.key,
                          r["state"], questions[r["lang"]])
            gold = gold_from_answers(
                questions[r["lang"]], ans.get("answers"))
            results[r["state_hash"]] = {
                "gold": gold,
                "confidence": confidence_of(gold),
                "input_tokens": (ans.get("usage") or {}).get("input_tokens"),
            }
        except Exception as exc:  # HTTPError, timeouts, bad payloads
            errors[r["state_hash"]] = f"{type(exc).__name__}: {exc}"
        if i % 25 == 0:
            (root / args.out).write_text(json.dumps(
                {"results": results, "errors": errors}))
            print(f"  evaluated {i}/{len(rows)}", flush=True)
    (root / args.out).write_text(json.dumps(
        {"results": results, "errors": errors}))
    print(f"done in {time.time()-t0:.0f}s "
          f"({len(results)} ok, {len(errors)} errors) -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
