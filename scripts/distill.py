#!/usr/bin/env python3
"""Distill gold judgment rows from a local laya-serve.

For each state: POST /v1/systemone with the routed checkpoint
(typed-decisions for en, multilingual for zh) and the language-matched
question set, then record the answer distributions as `gold` — the
same `{state, questions, gold}` schema laya's train.py consumes.

Usage:
    distill.py --states states.jsonl \\
        --questions-en data/questions_en_done_when.json \\
        --questions-zh data/questions_zh_done_when.json \\
        --out data/rows.jsonl [--api http://127.0.0.1:8000] [--key laya-local]
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def ask_jev(api: str, key: str, model: str, state: str, questions: dict) -> dict:
    payload = json.dumps(
        {"state": state, "model": model, "questions": questions}
    ).encode()
    req = urllib.request.Request(
        f"{api.rstrip('/')}/v1/systemone",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "x-api-key": key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def gate_questions(qfile: dict) -> dict:
    """Keep only real question entries — the gate's own rule
    (review_chain.load_questions): dict values carrying a
    "type". Top-level scalars (default_fail/default_warn) are
    client-side thresholds and make the server 422."""
    return {k: v for k, v in qfile.items()
            if isinstance(v, dict) and "type" in v}


def gold_from_answers(questions: dict, answers: dict) -> tuple[dict, dict]:
    """Map /v1/systemone answers to the finetune schema.

    Returns (gold, expected):
    - noul:   gold {"true": p, "false": 1-p}, expected p >= 0.5
    - choice: gold {"probabilities": {...}}, expected argmax
    """
    gold = {}
    expected = {}
    for qid, ans in (answers or {}).items():
        q = questions.get(qid, {})
        if q.get("type") == "noul" or ans.get("type") == "noul":
            p = float(ans.get("noul", 0.0))
            gold[qid] = {"probabilities": {
                "true": round(p, 6), "false": round(1.0 - p, 6)}}
            expected[qid] = bool(p >= 0.5)
        elif q.get("type") == "choice" or ans.get("type") == "choice":
            probs = ans.get("probabilities", {})
            gold[qid] = {"probabilities": probs}
            expected[qid] = max(probs, key=probs.get) if probs else None
    return gold, expected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", required=True)
    ap.add_argument("--questions-en", required=True)
    ap.add_argument("--questions-zh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="laya-local")
    args = ap.parse_args()

    q_en = gate_questions(json.loads(Path(args.questions_en).read_text(encoding="utf-8")))
    q_zh = gate_questions(json.loads(Path(args.questions_zh).read_text(encoding="utf-8")))

    rows = []
    errors = []
    for line in open(args.states, encoding="utf-8"):
        if not line.strip():
            continue
        st = json.loads(line)
        lang = st.get("lang", "en")
        questions = q_zh if lang == "zh" else q_en
        model = "multilingual" if lang == "zh" else "typed-decisions"
        try:
            resp = ask_jev(args.api, args.key, model, st["state"], questions)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            errors.append({"state_hash": st["state_hash"], "error": str(e)})
            continue
        gold, expected = gold_from_answers(questions, resp.get("answers"))
        if not gold:
            errors.append({"state_hash": st["state_hash"], "error": "no answers"})
            continue
        rows.append({
            "state": st["state"],
            "questions": questions,
            "gold": gold,
            "expected": expected,
            "state_hash": st["state_hash"],
            "lang": lang,
            "source": st.get("source", ""),
            "distilled_from": model,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({
        "rows": len(rows),
        "errors": len(errors),
        "by_lang": {l: sum(1 for r in rows if r["lang"] == l) for l in ("en", "zh")},
    }, indent=2))
    if errors:
        Path(str(out) + ".errors.json").write_text(
            json.dumps(errors, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
