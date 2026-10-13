#!/usr/bin/env python3
"""Expand the candidate pool and screen it with laya itself.

Stage 1  gen states: per-commit diffs from selected repos
         (lang = gate routing rule: Han char anywhere -> zh)
Stage 2  dedup: exact state_hash within the pool and against
         the existing 58 committed rows
Stage 3  laya evaluation: POST /v1/systemone with the routed
         checkpoint and the language-matched question set
Stage 4  filters (gold quality):
           error        HTTP/422 from the gate
           degenerate   state < 80 chars -- nothing to judge
           weak-verdict verdict max prob < 0.6 (near-uniform)
           weak-noul    |p - 0.5| < 0.15 (coin flip)
         flags (kept, marked):
           truncated    state hit the 30000-char cap (the
                        router judges the first 1024 tokens
                        either way -- gold and training see
                        the same view)
Stage 5  artifacts under data/screened/:
           kept.jsonl   training-schema rows + confidence meta
           out.jsonl    rejected states with reason + detail
           summary.json counts, per-reason, per-lang, per-repo

Usage: screen.py [--api http://127.0.0.1:8000] [--key laya-local]
                 [--workers 8] [--max-per-repo N]
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jevclient import (ask_jev, confidence_of, gate_questions,
                       gold_from_answers)

HAN = re.compile("[一-鿿]")
MAX_CHARS = 30000
MIN_STATE_CHARS = 80
VERDICT_MIN_MAX = 0.6
NOUL_MARGIN = 0.15

# Repos that add diverse, on-domain code-change states. The
# new-api family is skipped: one proxy codebase cloned four
# times, so its diffs are near-duplicates of each other.
# Caps sample the newest N first-parent commits (0 = all).
REPOS = [
    ("canon", 0), ("herdr", 1200), ("ferrite-legacy", 200),
    ("kymido", 200), ("mono", 0), ("ui-kit", 0),
    ("novel", 100), ("kime", 100), ("wildtoken", 150),
    ("dotfiles", 150), ("sightline", 0), ("silverq", 0),
    ("sentinel", 0), ("deskctl", 0), ("gugu", 0),
    ("algorchemy", 0), ("laya-canon-judgments", 0),
    ("oh-my-pi", 300), ("ferrite-ci", 0), ("ferrite-ci-pr", 0),
    ("ricq", 0), ("ainotation", 0),
    ("new-api", 1500), ("newapi-xiaocongyu", 1500),
    ("mnemo2u", 0),
]
PROJECTS = Path("/home/hathaway/projects")

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def gen_states(repo: Path, limit: int) -> list[dict]:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--first-parent", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    if limit > 0:
        out = out[:limit]
    rows = []
    for commit in out:
        diff = subprocess.run(
            ["git", "-C", str(repo), "show", "--format=",
             "--no-ext-diff", commit],
            capture_output=True, text=True, check=True,
            encoding="utf-8", errors="replace",
        ).stdout
        if not diff.strip():
            continue
        if len(diff) > MAX_CHARS:
            diff = diff[:MAX_CHARS]
        rows.append({
            "state_hash": hashlib.sha256(diff.encode()).hexdigest()[:16],
            "lang": "zh" if HAN.search(diff) else "en",
            "source": f"{repo.name}:{commit[:12]}",
            "state": diff,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="laya-local")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="data/screened")
    ap.add_argument("--resume",
                    default="data/screened/.eval-cache.json",
                    help="evaluated-states checkpoint; a rerun "
                         "resumes from it (empty disables)")
    ap.add_argument("--export-pending", default="",
                    help="write the not-yet-evaluated states to this "
                         "JSONL and exit (for CI evaluation)")
    ap.add_argument("--merge-fragment", default="",
                    help="merge a CI cache fragment into --resume and "
                         "exit")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.merge_fragment:
        if not args.resume:
            sys.exit("--merge-fragment needs --resume (cache path)")
        cache_path = root / args.resume
        frag = json.loads((root / args.merge_fragment).read_text())
        blob = (json.loads(cache_path.read_text())
                if cache_path.exists() else {"results": {}, "errors": {}})
        blob["results"].update(frag.get("results", {}))
        for h in frag.get("results", {}):
            blob["errors"].pop(h, None)
        blob["errors"].update(frag.get("errors", {}))
        cache_path.write_text(json.dumps(blob))
        log(f"merged {args.merge_fragment}: {len(blob['results'])} results, "
            f"{len(blob['errors'])} errors -> {args.resume}")
        return
    questions = {
        "en": gate_questions(json.loads(
            (root / "data/questions_en_done_when.json").read_text())),
        "zh": gate_questions(json.loads(
            (root / "data/questions_zh_done_when.json").read_text())),
    }

    # Stage 1: generate
    log("stage 1: generating candidate states")
    pool = []
    per_repo = {}
    for name, limit in REPOS:
        repo = PROJECTS / name
        if not (repo / ".git").exists():
            log(f"  skip {name}: no repo")
            continue
        rows = gen_states(repo, limit)
        per_repo[name] = Counter(r["lang"] for r in rows)
        pool.extend(rows)
        log(f"  {name}: {len(rows)} states "
            f"({dict(per_repo[name])})")
    log(f"pool: {len(pool)} states")

    # Stage 2: dedup
    existing = set()
    for line in (root / "data/rows.jsonl").read_text().splitlines():
        existing.add(json.loads(line)["state_hash"])
    seen = set()
    candidates = []
    dup_pool = dup_existing = 0
    for r in pool:
        if r["state_hash"] in existing:
            dup_existing += 1
            continue
        if r["state_hash"] in seen:
            dup_pool += 1
            continue
        seen.add(r["state_hash"])
        candidates.append(r)
    log(f"stage 2: dedup -> {len(candidates)} candidates "
        f"({dup_pool} in-pool dups, {dup_existing} already-trained)")

    # Stage 3: laya evaluation (threaded, checkpointed)
    cache = (root / args.resume) if args.resume else None
    results, errors = {}, {}
    if cache and cache.exists():
        blob = json.loads(cache.read_text())
        results = blob.get("results", {})
        errors = blob.get("errors", {})
    todo = [r for r in candidates
            if r["state_hash"] not in results
            and r["state_hash"] not in errors]
    log(f"stage 3: evaluating {len(todo)} states with laya "
        f"({args.workers} workers, "
        f"{len(candidates) - len(todo)} cached)")
    if args.export_pending:
        pend = [r for r in candidates
                if r["state_hash"] not in results]
        out = root / args.export_pending
        out.write_text("".join(
            json.dumps({k: r[k] for k in
                        ("state_hash", "source", "lang", "state")}) + "\n"
            for r in pend))
        log(f"exported {len(pend)} pending states -> {args.export_pending}")
        return
    done = [0]
    cache_lock = threading.Lock()

    def save_cache() -> None:
        if not cache:
            return
        with cache_lock:
            cache.write_text(json.dumps(
                {"results": results, "errors": errors}))

    def evaluate(r: dict) -> None:
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
        done[0] += 1
        if done[0] % 25 == 0:
            save_cache()
            log(f"  evaluated {done[0]}/{len(todo)}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(evaluate, todo))
    save_cache()
    log(f"stage 3: done in {time.time()-t0:.0f}s "
        f"({len(results)} ok, {len(errors)} errors)")

    # Stage 4: filters
    kept, out = [], []
    truncated = 0
    for r in candidates:
        h = r["state_hash"]
        if h in errors:
            out.append({**r, "reason": "error", "detail": errors[h]})
            continue
        if len(r["state"]) < MIN_STATE_CHARS:
            out.append({**r, "reason": "degenerate",
                        "detail": f"{len(r['state'])} chars"})
            continue
        res = results[h]
        conf = res["confidence"]
        weak = []
        if conf.get("verdict", 1.0) < VERDICT_MIN_MAX:
            weak.append(f"verdict max={conf['verdict']:.3f}")
        if conf.get("evidence_sufficient", 0.5) < NOUL_MARGIN:
            weak.append(f"noul |p-0.5|={conf['evidence_sufficient']:.3f}")
        if weak:
            out.append({**r, "reason": "weak-gold",
                        "detail": "; ".join(weak),
                        "confidence": conf})
            continue
        if len(r["state"]) >= MAX_CHARS:
            truncated += 1
        row = {
            "state": r["state"],
            "questions": questions[r["lang"]],
            "gold": res["gold"],
            "state_hash": h,
            "lang": r["lang"],
            "source": r["source"],
            "screened": {
                "confidence": res["confidence"],
                "input_tokens": res["input_tokens"],
                "truncated_at_cap": len(r["state"]) >= MAX_CHARS,
            },
        }
        kept.append(row)

    # Stage 5: artifacts
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "kept.jsonl").open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "out.jsonl").open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    reasons = Counter(r["reason"] for r in out)
    kept_lang = Counter(r["lang"] for r in kept)
    out_lang = Counter(r["lang"] for r in out)
    kept_repo = Counter(r["source"].split(":")[0] for r in kept)
    conf_verdict = [r["screened"]["confidence"]["verdict"]
                    for r in kept if "verdict" in r["screened"]["confidence"]]
    summary = {
        "pool_states": len(pool),
        "per_repo_generated": {k: dict(v) for k, v in per_repo.items()},
        "candidates_after_dedup": len(candidates),
        "dup_in_pool": dup_pool,
        "dup_already_trained": dup_existing,
        "kept": len(kept),
        "rejected": len(out),
        "reject_reasons": dict(reasons),
        "kept_by_lang": dict(kept_lang),
        "rejected_by_lang": dict(out_lang),
        "kept_by_repo": dict(kept_repo),
        "kept_truncated_at_cap": truncated,
        "kept_verdict_confidence": {
            "min": min(conf_verdict) if conf_verdict else None,
            "p50": sorted(conf_verdict)[len(conf_verdict) // 2]
                   if conf_verdict else None,
            "mean": (sum(conf_verdict) / len(conf_verdict))
                    if conf_verdict else None,
        },
        "existing_rows": len(existing),
        "total_rows_if_added": len(existing) + len(kept),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log("stage 5: artifacts written to " + str(out_dir))
    log(json.dumps({k: v for k, v in summary.items()
                    if k != "per_repo_generated"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
