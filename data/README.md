# Data format

## rows JSONL (one line per judgment state)

```jsonl
{"state": {...}, "questions": {"q1": {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}}}, "gold": {"q1": {"true": 0.9, "false": 0.1}}, "state_hash": "ab12cd34", "lang": "en"}
```

| field | meaning |
|---|---|
| `state` | evidence payload (diff / propositions / ticket text) — what the model reads |
| `questions` | typed questions: id → `{type, instructions, criteria}` (Jev question schema) |
| `gold` | per-question **probability distribution** over the criteria options — RLCD needs distributions; hard labels become one-hot |
| `state_hash` | group key for splitting (all rows of one state share it) |
| `lang` | `en` / `zh` — used for balance audits (target 50/50) |

## Rules

- `gold` distributions must sum to 1 per question
- one `state_hash` = one group; never split a group across train/validation/test
- raw sources (logs, dumps) stay out of git; only processed rows + splits manifests are committed
