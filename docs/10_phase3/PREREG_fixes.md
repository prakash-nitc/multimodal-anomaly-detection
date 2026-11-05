# Pre-registration: four fixes, two encoders

Written 5 November 2026, before any of these conditions was scored.
Code: `notebooks/fixes_sweep.py`, `da_zvad/prompts/fixes.py`.

## Why

The normal-side run (`PREREG_normal_side.md`, commit a87ff87) found no system that
beats the plain detector across datasets. Reading the related work afterwards
showed a mistake in that design: **our near-miss component (N) adds look-alike
actions to the normal side, while NOVA, which motivated it, forbids them**
(arXiv 2609.06360, §3.2 and supplement F.3). That run therefore did not test
NOVA's idea, and the reason given for its prediction P1 was wrong. The run's rule
also tested a component by removing it from the full stack, and so kept N even
though N alone lowers UCF-Crime and XD-Violence on ViT-L.

The PE-Core run then showed that several effects depend on the encoder. This run
tests four inexpensive fixes on both encoders.

## What is already known

This is not a blind experiment, and it cannot show generalisation.

- All four datasets have shaped the design. Results for no sentence, the shared
  sentence, LLaVA, retrieval and the five old components are known on both
  encoders. None of the conditions below has been scored.
- The fixes come from papers whose results use UCF-Crime and XD-Violence.
- The confidence intervals are not corrected for multiple comparisons.
- Results here are exploratory. Only the sealed test can confirm them.

## The four fixes

Each is one switch, is training-free and needs no generative model at run time.
All use the cached frame embeddings; no image is re-encoded. λ stays 100.

| | Fix | Definition | Source |
|---|---|---|---|
| **C** | Calm normal side | Add eight fixed phrases to the normal side: four calm anchors (`banks.CALM_ANCHORS`) and "a person standing still", "a person sitting quietly", "two people talking calmly", "a person walking slowly". The same list for every dataset, written without any class list. | NOVA's rules, not its generated text |
| **A** | Visual anchor | The unit's five calm frames (`retrieval.calm_indices`) give a mean image vector *a*. Normal similarity becomes (1 − w)⟨f, pₙ⟩ + w⟨f, a⟩, with w = 1/11. | NOVA eq. 4–7; 1/11 is its anchor's share of a top-50% mean over 21 items |
| **K** | Side keywords | Prefix every normal phrase with "Normal scene: " and every abnormal phrase with "Anomalous scene: ", sentences included. | NOVA p.6; idea from Flashback's repulsive prompting |
| **S** | Anomaly weight | Margin becomes α⟨f, pₐ⟩ − normal similarity, with α = 0.95. | Flashback p.9 |

Differences from the sources:
- A uses our existing five calm frames, not NOVA's soft weights over 32.
- Scoring stays prototype-based, not per-class top-50%.
- K uses NOVA's prefixes, not Flashback's templates.

Secondary, outside the decision: α = 0.90 (NOVA's value, with a bootstrap) and w = 1/4
(point estimate only).

## Conditions

For each encoder (ViT-L/14 and PE-Core-L-14-336) and dataset (ShanghaiTech,
UCF-Crime, XD-Violence, NWPU Campus):

- **Table:** all 16 combinations of the fixes × 4 sentence conditions (none, shared,
  LLaVA, retrieved), as point estimates.
- **Bootstrapped (1000 video-level paired resamples, seed 20261104):**
  - each fix and all four (CAKS) against the plain detector, plus α = 0.90;
  - shared, LLaVA and retrieved against the plain detector and each other;
  - CAKS with each sentence against that sentence alone;
  - retrieved and LLaVA against shuffled controls.
- **Audit** (level/shape, `measurement_audit`): each fix, CAKS and each sentence
  against the plain detector.

**Correction to the earlier run:** that run's shuffled controls used 20 shuffles for
the point estimate and 5 in the bootstrap, so a point could fall outside its own
interval. Here the same 5 shuffles give both. Earlier shuffled-control intervals
should not be quoted.

## Gates

For each dataset and encoder, before anything else is reported:
1. the new margin path with all fixes off matches the legacy path (max difference ≤ 1e-5);
2. no sentence, shared, LLaVA and retrieved reproduce the earlier run's values for
   that encoder (difference ≤ 1e-5).

A dataset that fails is not reported. A failed gate stops `decide`.

## Decision rule

Fixed now, separately for each encoder. A fix **passes** if, against the plain
detector with no sentence:
- the point estimate is above 0 on more than half of the datasets;
- the 95% CI lies above 0 on at least 2;
- the 95% CI lies below 0 on none.

**Combination check:** the passing fixes are then scored together. While the
combination has a CI below 0 on any dataset, or is positive on no more than half,
the passing fix with the smallest mean gain is dropped and the rest are re-scored.
If none passes, that encoder's candidate is the plain detector, and that is the
result.

**Leave one dataset out:** the rule is applied to the other three datasets (with
"more than half" meaning at least 2) and its choice is scored on the hidden one.
This checks the rule only, because the fixes were chosen knowing all four.

**Encoders:** PE-Core minus ViT-L, paired on the same videos, for no sentence, shared,
LLaVA, retrieved and CAKS. This run does not choose an encoder. The freeze addendum does,
or carries both to the sealed test.

## Predictions

| | Prediction | Why |
|---|---|---|
| F1 | C beats the old N on UCF-Crime on both encoders, and on XD-Violence on ViT-L | N added look-alikes |
| F2 | A changes UCF-Crime and XD-Violence by under 0.03 on both encoders | NOVA's anchor alone gained at most 1.37 points |
| F3 | K gains under +0.05 on UCF-Crime on both encoders | Our two sides are already worded differently |
| F4 | S is calibration-dominated or has no effect on at least 3 of 4 datasets, on both encoders | It shifts one side's similarity |
| F5 | At most one fix passes the rule on both encoders | Effects have varied by dataset and encoder so far |
| F6 | LLaVA stays below no sentence on XD-Violence on both encoders, with and without all fixes | It did on both encoders, and no fix addresses contaminated captions |
| F7 | Some fix has CIs on opposite sides of 0 on the two encoders on one dataset | Worries did on ShanghaiTech |
| F8 | Each unit's own LLaVA sentence beats another unit's on at least 3 of 4 datasets, on both encoders | LLaVA's sentence carried scene information on UCF-Crime and ShanghaiTech |

Honest expectation: C helps on some datasets; A, K and S are small; no single setup clearly
beats the plain detector everywhere.

## Limitations

- A uses each test video's or camera's opening frames, which is transductive. In
  deployment these would be setup footage.
- The eight calm phrases are one wording. Other wordings would give other numbers.
- λ = 100 suits ViT-L's scale and PE-Core's learned 99.9, but other settings are untested.
- A puts an image vector, much closer to frames than any text, into a margin built for text.
  The run records each condition's tied-score fraction so saturation can be seen.
- Avenue has no cached embeddings on either encoder and is not included.

## Next

Step 3 (classmates) and step 4 (patch scoring) have their own written rules.
Then a dated addendum freezes the candidates for the single sealed run.

## Run

```bash
cd ~/dazvad/repo && git pull && source ~/dazvad/venv/bin/activate
python notebooks/fixes_sweep.py selftest
python notebooks/fixes_sweep.py run --backbone vitl 2>&1 | tee ~/dazvad/logs/fixes_vitl.log
python notebooks/fixes_sweep.py run --backbone pe-l 2>&1 | tee ~/dazvad/logs/fixes_pe.log
python notebooks/fixes_sweep.py decide 2>&1 | tee ~/dazvad/logs/fixes_decide.log
```

## Addenda

None yet. Changes after this commit go here, dated, and never in the text above.
