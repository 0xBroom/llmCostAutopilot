# Annotator agreement — rubric ambiguity: two independent LLM annotators, fresh contexts, same model, taxonomy v1

- Generated: 2026-08-20
- Pass A: `pass-rubric-a`
- Pass B: `pass-rubric-b`
- Prompts: 25, from `config/calibration_prompts.yaml`

| Metric | Value |
| :-- | :-- |
| Cohen's κ | **0.940 (almost perfect)** |
| Raw agreement | 24 / 25 (96.0%) |
| Expected agreement by chance | 33.3% |

κ is reported, never gated — see `docs/complexity-taxonomy.md` section 9.

## Confusion matrix

Rows are pass A, columns are pass B.

| A \ B | 1 | 2 | 3 |
| :-- | --: | --: | --: |
| **1** | 8 | 0 | 0 |
| **2** | 0 | 8 | 0 |
| **3** | 0 | 1 | 8 |

## Disagreements

Each of these owes the rubric a clarification (section 10).

### `cal-21` — tier 3 vs 2

- **pass-rubric-a** (tier 3): The underlying task is a tier-2 one-sentence review condensation, escalated one tier because the material is reasoned over in Portuguese.
- **pass-rubric-b** (tier 2): Summarising the review to one sentence is ordinary condensation; Portuguese is not a low-resource language, so the multilingual escalation ruling does not apply here.
