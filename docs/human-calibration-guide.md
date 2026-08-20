# Running the human calibration (κ_human)

`docs/complexity-taxonomy.md` has one number missing, and it is the important
one. This guide is how you produce it.

It costs about **90 minutes of your attention, split across three days** — one
labelling session, a 48-hour gap you must not skip, a second labelling
session, and one command. At the end you will have a defensible answer to the
question every reviewer of this project will ask: *"how do you know your
labels are consistent?"*

---

## Quick path

| Day | Action | Time |
| :-- | :-- | :-- |
| 1 | Label all 25 prompts into `pass-human-a.yaml` | ~40 min |
| 1 | Commit it, then **do not open it again** | 1 min |
| 3+ | Label the same 25 prompts from scratch into `pass-human-b.yaml` | ~40 min |
| 3+ | Run the κ CLI, read the disagreements | 5 min |
| 3+ | Append a clarification per disagreement, update the version table | ~15 min |

The exact commands are in [step-by-step](#step-by-step). Read the next two
sections first — the procedure only means something if you understand what it
is measuring.

---

## What κ_human actually is

**Cohen's κ answers one question: how much of the agreement between two
labellers is more than luck?**

Raw agreement alone lies. Suppose you label 25 prompts and 20 of them come out
tier 2 both times. That is 80% agreement — and almost worthless, because a
labeller who wrote "tier 2" on every line without reading anything would also
score around 80%. Chance agreement has to be subtracted out.

```
        observed agreement  −  agreement expected by chance
  κ  =  ────────────────────────────────────────────────────
                    1  −  agreement expected by chance
```

`p_e`, the chance term, comes from how often each labeller used each tier.
Worked from our own rubric run:

| Term | Value | Where it came from |
| :-- | :-- | :-- |
| `p_o` (observed) | 0.960 | agreed on 24 of 25 prompts |
| Tier usage, pass A | 8 / 8 / 9 | tiers 1 / 2 / 3 |
| Tier usage, pass B | 8 / 9 / 8 | tiers 1 / 2 / 3 |
| `p_e` (chance) | 0.333 | `(8/25)(8/25) + (8/25)(9/25) + (9/25)(8/25)` |
| **κ** | **0.940** | `(0.960 − 0.333) / (1 − 0.333)` |

Two properties worth internalising:

- **κ = 0 means "no better than a coin"**, not "no agreement". κ can go
  negative, which means the two passes disagree *systematically* — worse than
  random.
- **A balanced tier distribution makes κ harder to score well on.** Our
  `p_e` of 0.333 is the honest, difficult case. If you had labelled everything
  tier 2, `p_e` would approach 1.0 and κ would collapse toward undefined. The
  statistic punishes a lazy label distribution, which is exactly what you want
  from it.

### The "human" part is the whole point

We already have a κ. It is 0.940, and it does **not** count as κ_human. It was
produced by two LLM annotators reading the rubric cold in separate contexts,
and it measures one thing only: *is this document written unambiguously enough
that two independent readers apply it the same way?*

That is a useful smoke test and a weak one:

| | κ_rubric (done) | κ_human (this guide) |
| :-- | :-- | :-- |
| Who labelled | two instances of one model | you, twice, 48h apart |
| What it measures | document ambiguity | whether the labels behind #11 are reproducible |
| Known bias | **upward** — two instances of one model share priors two people never would | downward on day 3 if you are tired; upward from your own memory |
| Belongs in the README headline | no | yes |
| Feeds #11's dataset | never | never — see step 6 |

#11's rule 1 forbids LLM-generated labels in the dataset, so no LLM pass can
ever stand in for this. The dataset's credibility rests on a human having
labelled it consistently, and κ_human is the only evidence of that.

### One annotator is allowed. Overselling it is not

The textbook version of this uses two people. You are one person, so you
substitute yourself-in-48-hours for the second annotator. **That is a
legitimate substitute if and only if you describe it as what it is.**

Say "single annotator, blind self-relabel after 48 hours". Never say
"double-labelled" — that implies a second person, and a reviewer who spots the
gap between your words and your procedure will stop trusting every other
number in the repo. The taxonomy document, the README and the model registry
all already use the honest phrasing; keep it.

**Why 48 hours:** you are not testing your memory, you are testing the
*rubric's grip on you*. Same-day relabelling measures recall — you will
reproduce your earlier answers without re-deriving them, and κ comes out near
1.0 meaning nothing. Two days is enough for the specific prompts to fade while
the rubric stays learned.

**A disclosure you owe the reader:** you wrote this rubric. A blind self-relabel
cannot remove the fact that you already hold its reasoning in your head, which
inflates κ_human relative to a stranger applying the document. Say so in one
sentence next to the number. If you want the stronger measurement, hand a
colleague `docs/complexity-taxonomy.md` and `config/calibration_prompts.yaml`
— nothing else, no explanation from you — and use their labels as pass B.
Then it is genuinely two annotators and you can say so.

---

## Step-by-step

### Step 0 — Do not read these files first

Three files in this repo already contain labels for the same 25 prompts:

```
artifacts/calibration/2026-08-20/pass-rubric-a.yaml
artifacts/calibration/2026-08-20/pass-rubric-b.yaml
artifacts/calibration/2026-08-20/agreement.md
```

Reading any of them before you finish **both** of your own passes destroys the
measurement, and nothing in the output will reveal that it happened. Same for
section 10 of the taxonomy, which names the prompts that were argued about.

If you have already read them recently, wait a week or swap in fresh prompts.
An honest "not yet run" beats a contaminated 0.9.

### Step 1 — Set up pass A

```bash
export CAL_DATE=$(date +%F)      # the date you START; used for the whole run
mkdir -p "artifacts/calibration/$CAL_DATE"
cp config/calibration_prompts.yaml "artifacts/calibration/$CAL_DATE/pass-human-a.yaml"
```

Then add the same dated directory to `.gitignore`'s re-include block, next to
the existing runs — `artifacts/*` is ignored by default and each committed run
is re-included explicitly:

```gitignore
!artifacts/calibration/<your-date>/
```

### Step 2 — Label, following section 8

Open `pass-human-a.yaml` and fill in four fields per prompt. Follow
[section 8 of the taxonomy](complexity-taxonomy.md) literally, in order — the
failure mode is reaching for a tier first and rationalising it after.

```yaml
  - id: cal-01
    tier: 1
    label_reason: "Both fields are present in the receipt; the task is to locate and copy them."
    signals: [transformation]
    edge_case: null
```

| Field | Rule |
| :-- | :-- |
| `tier` | 1, 2 or 3. Nothing else parses. |
| `label_reason` | Mandatory, one sentence, phrased as *what the answer requires* plus the signal that escalated it. This is the field you will thank yourself for in three weeks. |
| `signals` | **Every** applicable slug, escalating one first. Vocabulary is in section 3. |
| `edge_case` | The matching slug from section 6, or `null`. |

Two habits that keep the pass honest:

- **When two tiers are both defensible, label the lower one** and put the
  argument for the higher one in `label_reason`. The system can escalate
  under-routing at runtime; it has no way to detect money wasted on
  over-routing.
- **Do not consult a model.** Not for a tie-break, not "just to check". That
  is the line #11 rule 1 draws.

Commit the file when the last prompt is done. Committing is what makes "I did
not touch it afterwards" verifiable rather than a claim.

### Step 3 — Wait 48 hours

Not 24. Do something else.

### Step 4 — Pass B, from the unlabelled original

Start from `config/calibration_prompts.yaml`, **never** from a copy of pass A:

```bash
cp config/calibration_prompts.yaml "artifacts/calibration/$CAL_DATE/pass-human-b.yaml"
```

Label all 25 again. Do not open pass A, do not open `agreement.md`, do not
skim your old reasons. If you find yourself trying to remember what you wrote
last time, stop and re-derive from the rubric instead — remembering is the one
thing this pass must not measure.

### Step 5 — Compute it

```bash
uv run python -m autopilot.interfaces.cli.kappa \
  --pass-a "artifacts/calibration/$CAL_DATE/pass-human-a.yaml" \
  --pass-b "artifacts/calibration/$CAL_DATE/pass-human-b.yaml" \
  --label "human blind self-relabel, 48h apart, single annotator (rubric author)" \
  --markdown "artifacts/calibration/$CAL_DATE/agreement-human.md"
```

`--label` is required on purpose: it is the sentence a reader uses to judge
whether the number means anything. Describe the procedure, not the intent.

Expect one of these:

| Output | What it means | What to do |
| :-- | :-- | :-- |
| `Cohen's kappa = 0.7xx (substantial)` | Normal, healthy result for a real rubric | Report it. Do not try to improve it. |
| `Cohen's kappa = 0.9xx` | Suspiciously high for a human pair | Check you really started pass B from the unlabelled original. If you did, say the number and note the memory caveat. |
| `Cohen's kappa = 0.4xx (moderate)` | The rubric has real holes | Report it, then fix the holes it named. That is a finding, not a failure. |
| `undefined` | You used one tier for all 25 | The calibration set spans all three tiers; re-read section 2's boundary questions. |
| A `ValueError` and no number at all | A refusal, not a result: an unlabelled row, mismatched prompt ids between the passes, or a tier outside 1..3 | The message names the offending prompt ids. Fix the worksheet and re-run. |

### Step 6 — Turn each disagreement into a rubric change

This is the step people skip, and it is where the value is. κ is one number
for a README; the disagreement list is the thing that makes the *next*
calibration better.

For each entry in the report's "Disagreements" section, append to
[section 10](complexity-taxonomy.md) of the taxonomy:

1. The prompt id and the two tiers.
2. Both reasons — you will have written them from two different mental states,
   which is genuinely informative.
3. **A ruling that removes the judgement**, not one that restates it. The
   v1 run's `cal-21` split on "is Portuguese low-resource?"; the fix replaced
   the judgement with a declared list of languages. A clarification that says
   "use your best judgement about the language" would have changed nothing.

Then:

- Add the κ_human column value to the **version table** in section 11.
- Replace the *"not yet run"* wording in section 9 with the result, the
  annotator description, and the memory-contamination disclosure.
- Update the README paragraph that currently says the human number is missing.
- Optionally extend `tests/unit/test_kappa.py`'s
  `test_recorded_kappa_is_rederivable_from_the_committed_passes` to pin the
  human run too, so the number in the document stays a computation over
  committed files rather than an assertion.

### Step 7 — Decide about the rubric version

Section 11's rule: **a changed ruling bumps `version`; clarified prose does
not.**

If a clarification from step 6 changes *when* a ruling applies — as the
`cal-21` language list did — then strictly the document has moved and the κ you
just measured describes the text as it stood before. Handle it the way v1 did:
record the commit hash the calibration measured, and say plainly that the next
calibration measures the clarified text. Do not silently re-attribute the
number to the new version.

---

## What invalidates the whole thing

Four ways to waste the 90 minutes. All of them are invisible in the output.

| Mistake | Why it ruins the number |
| :-- | :-- |
| Peeking at pass A, the LLM passes, or section 10 before finishing pass B | Measures recall instead of the rubric's grip |
| Relabelling until κ crosses a threshold | κ optimised against is no longer an estimate of anything — this is exactly why the project reports and never gates |
| Relabelling the same day | Same as peeking, slower |
| Calling it "double-labelled" | The procedure and the description stop matching, and a reader who notices discounts everything else you measured |

If any of these happens, the honest move is to say so in the report's
`--label` text or to redo the run with fresh prompts. A missing number is
recoverable; a number nobody can trust is not.

---

## Checklist

- [ ] Neither LLM pass, `agreement.md`, nor taxonomy section 10 was read since
      before pass A
- [ ] `pass-human-a.yaml` committed on day 1, untouched after
- [ ] At least 48 hours elapsed
- [ ] `pass-human-b.yaml` started from `config/calibration_prompts.yaml`
- [ ] All 25 prompts labelled in both passes, `label_reason` filled in every row
- [ ] κ computed with a `--label` that describes the real procedure
- [ ] `agreement-human.md` and both passes committed, dated directory
      re-included in `.gitignore`
- [ ] One clarification appended per disagreement, each removing a judgement
- [ ] Taxonomy section 9, section 11 version table, and the README updated
- [ ] The phrase "single annotator, blind self-relabel" appears wherever the
      number does

---

## Next step

With κ_human recorded, #10 is closed and **#11 is unblocked**: 200+ labelled
prompts using this same procedure and vocabulary, carrying
`taxonomy_version: 1`. The calibration you just ran is the evidence that
labelling 200 of them will produce something a classifier can learn from
rather than a transcript of one afternoon's mood.
