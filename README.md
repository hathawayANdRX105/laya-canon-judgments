# laya-canon-judgments

Fine-tune **laya-multilingual** (one model, en+zh) for canon-spec judgments:
the typed questions the canon gate uses (`classify` / `compare` / `extract` /
`rerank` / `find`), in both English and Chinese.

## Why one model

laya-multilingual is a 100+ language encoder (mmBERT-base) with en and zh in
pretraining. A joint en+zh decision fine-tune is standard multilingual
practice. Tradeoff vs the current 3-checkpoint + router setup: ~10 points of
English decision accuracy (multilingual base 0.657 vs english checkpoint
0.783 on MASSIVE) in exchange for one ~1.3GB checkpoint instead of three
resident (~3.9GB) and no routing.

## Layout

| Path | Purpose |
|---|---|
| `data/` | rows JSONL format spec + splits (generated, gitignored) |
| `scripts/make_splits.py` | group-aware train/val/test splitter (fixed seed) |
| `scripts/prepare_rows.py` | build rows JSONL from raw sources (distillation + logs) |
| `notebooks/` | Kaggle notebook: SFT head + RLCD calibration |
| `docs/training-plan.md` | full plan: data, splits, recipe, Kaggle, eval gates |
| `results/` | eval reports (accuracy / ECE / option-order flip) |

## Quickstart

```bash
# split BEFORE any training (group-aware, fixed seed)
python scripts/make_splits.py data/rows.jsonl --seed 42 --out-dir data/splits
```

Row construction (distillation from the existing checkpoints) and the
training itself run on the laya toolchain (`train_cli.py` SFT head,
`train.py` RLCD) — see [docs/training-plan.md](docs/training-plan.md)
§3 and §5–6 for the recipe and the Kaggle notebook workflow.

## Status

- [x] repo + split tooling
- [x] seed dataset: 58 distilled states (28 en repo diffs + 30 zh)
- [x] head-only RLCD stage (local GPU) + held-out test eval
- [ ] full encoder RLCD (needs >=16GB GPU -- Kaggle 2xT4)
- [ ] scale dataset to thousands of states
- [ ] temperature refit on validation (zh ECE target < 0.15)
- [ ] final test eval (accuracy / ECE / flip rate)
