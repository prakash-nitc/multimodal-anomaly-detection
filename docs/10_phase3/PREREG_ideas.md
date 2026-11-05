# Pre-registration: the improvement round, one idea at a time

Started 5 November 2026. Code: `notebooks/ideas_sweep.py`, `da_zvad/prompts/ideas.py`.

## Why

The fixes run (`PREREG_fixes.md`, commit 9ae7671) found that eight calm phrases on the
normal side (C) improve PE-Core on three of four datasets, and passes the rule there.
Fix C does nothing clear on ViT-L, and the run used one wording. The related work
suggests further ideas that need no training.

## How the round works

- Ideas are tried one at a time, in the order below, starting with those most likely to work.
- Each step gets its own dated section here, committed before that step is scored.
- A step is judged against the current best, on PE-Core unless its section says otherwise.
- An idea that fails is dropped and not reworded for another attempt.
- At the end one model is chosen and frozen in a dated addendum. It then goes to the sealed test, once.
- Everything tried is reported, including failures.

Planned order: (1) check the current best; (2) both encoders together; (3) NOVA's and
Flashback's ideas: phrase bank, scoring by anomaly type, different templates per side;
(4) phrases chosen by scene, only if the bank is useful; (5) camera memory; (6) per-video
baseline. A step may be skipped if an earlier result makes it pointless; the reason goes here.

## What is already known

All four datasets have shaped the design, and many conditions have already been scored:
64 cells per encoder in the fixes run alone. Confidence intervals are not corrected for
multiple comparisons. Results here are exploratory, and only the sealed test can confirm them.

The rule, unless a section says otherwise, is the one fixed in `PREREG_fixes.md`: above 0 on
more than half of the datasets, 95% CI above 0 on at least 2, and CI below 0 on none.
Bootstraps use 1000 paired video-level resamples, seed 20261104.

---

## Step 1: check the current best (written 5 November 2026, before scoring)

### Questions

1. Is C a real effect or one lucky wording?
2. Does K earn its place on top of C? C + K looked best in the fixes table, but it was noticed
   afterwards and has no confidence intervals.
3. How does the new model compare with the old ones on the same videos?

### Conditions (PE-Core, no sentence)

| Name | Definition |
|---|---|
| base | Plain PE-Core |
| C | The eight calm phrases from `PREREG_fixes.md` |
| CK | C with the side keywords |
| C2 | A second wording, `ideas.CALM_SECOND`: "a deserted pavement", "an unoccupied hallway", "a vacant plaza", "a still courtyard with no one in it", "someone waiting patiently", "a person resting on a bench", "a few people chatting quietly", "someone strolling at an easy pace" |
| C2K | C2 with the side keywords |
| old_llava | ViT-L with the LLaVA sentence, from the fixes run's saved scores |
| old_plain | Plain ViT-L, from the same scores |

C2 has the same shape as C, four quiet places and four calm people, and no phrase in common
with it. Both halves of each wording are also scored alone, as point estimates only, to help
explain any effect. They are outside the decision.

### Gates

For each dataset, before anything is reported:
1. the calm phrases give identical margins through the new code path;
2. base, C and CK match the fixes run's table (difference ≤ 1e-5);
3. the videos match the ViT-L run, and its saved LLaVA and plain scores give that run's values.

A dataset that fails is not reported, and no decision is made.

### Decision rule

- **Wording:** C counts as real only if C2 passes the rule against base. If it fails, C is treated
  as one wording's result, and the current best goes back to plain PE-Core.
- **Keywords:** K is adopted only if CK passes the rule against C, and C2K is above C2 on more
  than half of the datasets.
- **Current best:** base if the wording check fails; otherwise CK if K is adopted; otherwise C.
  C stays the named wording because it was registered first.
- **Against the old models**, for base, C and CK against old_llava and old_plain: "better" if the
  rule passes; "no worse" if no CI is below 0; otherwise "worse". Only these words may be used.

### Predictions

| | Prediction | Why |
|---|---|---|
| G1 | C2 passes the rule against base | Calm content should matter more than the exact words |
| G2 | C2 and C differ within noise on at least 3 of 4 datasets | Same reason |
| G3 | K passes the rule on top of C | CK was above C on three datasets, though this was seen afterwards |
| G4 | C2 also lowers UCF-Crime | C did, and the cause should be the content |
| G5 | C beats old_llava on XD-Violence with a CI above 0, and no CI is below 0 | XD went from 0.417 to 0.635; the other three gaps are under 0.01 |

### Limitations

- I wrote the second wording knowing that C worked, so it is not independent of that knowledge.
- Two wordings show less than many would. Passing shows that C isn't tied to one list, not that all calm phrases work.
- The run is on PE-Core only, because C did nothing clear on ViT-L.

### Run

```bash
cd ~/dazvad/repo && git pull && source ~/dazvad/venv/bin/activate
python notebooks/ideas_sweep.py selftest
python notebooks/ideas_sweep.py step1 2>&1 | tee ~/dazvad/logs/ideas_step1.log
```

---

## Step 2: both encoders together (written 5 November 2026, before scoring)

### Question

Does averaging the two encoders beat PE-Core + C, the current best after step 1?
The two encoders are strong in different places, and neither needs training.

### Conditions (no sentence)

| Name | Definition |
|---|---|
| C | PE-Core + C, the current best |
| E | The mean of PE-Core + C's margin and plain ViT-L's margin, per frame |
| EC | The mean of PE-Core + C's margin and ViT-L + C's margin, per frame |
| old_llava, old_plain | As in step 1 |

Margins are averaged before the sigmoid and smoothing. Both encoders have a learned logit
scale close to 100 (99.9 and 100.0), so both margins are in the same units and neither
is rescaled. No weight is tuned.

### Gates

For each dataset: both encoders have the same videos, frame counts and labels; and PE-Core + C,
plain ViT-L, ViT-L + C and ViT-L with LLaVA match the fixes run (difference ≤ 1e-5).

### Decision rule

- E and EC are each judged against C by the rule.
- If one passes, it becomes the current best. If both pass, the one with the larger mean
  gain does. If neither passes, PE-Core + C stays.
- Against the old models, the same three words as step 1 apply.

### Predictions

| | Prediction | Why |
|---|---|---|
| H1 | At least one combination passes the rule against C | Combining models that fail in different places usually helps |
| H2 | EC is above C on XD-Violence | ViT-L was stronger than PE-Core on XD before any fix |
| H3 | Neither combination has a CI above 0 on UCF-Crime | ViT-L is much weaker than PE-Core on UCF |
| H4 | Both combinations are above C on NWPU Campus | Both encoders are close there, and their errors may differ |

### Limitations

- A combination doubles the image and text encoding at run time, a real cost in deployment. A pass
  shows a gain in score, not that the gain is worth that cost.
- Equal weights are one choice; other weights are untested, on purpose.

### Run

```bash
cd ~/dazvad/repo && git pull && source ~/dazvad/venv/bin/activate
python notebooks/ideas_sweep.py selftest
python notebooks/ideas_sweep.py step2 2>&1 | tee ~/dazvad/logs/ideas_step2.log
```

## Addenda

Changes to a step after its commit go here, dated, and never in that step's text.

**5 November 2026, step 1 outcome.** All gates passed at commit 7168814. The second wording
passes the rule, so C counts as real. K fails: it gains significantly on three datasets, but its
CI on NWPU Campus is below 0 (−0.014 [−0.026, −0.003]). The current best is PE-Core + C.
Predictions G1, G4 and G5 held; G2 and G3 were refuted.
