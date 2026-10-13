#!/usr/bin/env python3
"""Shared Jev-compatible client helpers for the screening pipeline.

Kept UI-free so both the local screener (`screen.py`) and the CI
evaluator (`eval_pending.py`) call the server with identical shapes.
"""

import json
import time
import urllib.error
import urllib.request


def ask_jev(api: str, key: str, state: str, questions: dict) -> dict:
    payload = json.dumps(
        {"state": state, "model": "auto", "questions": questions}
    ).encode()
    for attempt in range(3):
        req = urllib.request.Request(
            f"{api.rstrip('/')}/v1/systemone", data=payload,
            headers={"Authorization": f"Bearer {key}",
                     "x-api-key": key,
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 503 and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise


def gate_questions(qfile: dict) -> dict:
    return {k: v for k, v in qfile.items()
            if isinstance(v, dict) and "type" in v}


def gold_from_answers(questions: dict, answers: dict) -> dict:
    gold = {}
    for qid, ans in (answers or {}).items():
        if ans.get("type") == "noul":
            p = float(ans.get("noul", 0.0))
            gold[qid] = {"probabilities": {
                "true": round(p, 6), "false": round(1.0 - p, 6)}}
        elif ans.get("type") == "choice":
            gold[qid] = {"probabilities": ans.get("probabilities", {})}
    return gold


def confidence_of(gold: dict) -> dict:
    """Per-question gold confidence: max prob for choice,
    distance from the coin flip for noul."""
    out = {}
    for qid, dist in gold.items():
        probs = dist.get("probabilities", {})
        if not probs:
            out[qid] = 0.0
        elif len(probs) == 2 and set(probs) == {"true", "false"}:
            out[qid] = abs(probs["true"] - 0.5)
        else:
            out[qid] = max(probs.values())
    return out
