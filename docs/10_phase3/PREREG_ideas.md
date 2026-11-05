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

---

## Step 3: the papers' ideas (written 5 November 2026, before scoring)

### Question

NOVA and Flashback score higher than we do, mostly through better text and scoring. Do their
ideas help PE-Core + C, across crime and campus benchmarks, with one setup for all of them?

### Conditions (PE-Core, no sentence; normal side is always base + C)

| Name | Definition | Source |
|---|---|---|
| C | Current best | Step 1 |
| B | Abnormal side replaced by a bank of 48 concrete actions, 12 types of 4 (`ideas.BANK`), mean-pooled as now | NOVA and ASK-Hint: rich, concrete text |
| T | The same bank, scored by type: for each type the mean of its top 2 phrases, then the highest type, minus the mean of the top half of the normal phrases | NOVA eq. 4 and the max over classes, p.5 |
| D | Different templates per side on the existing phrases: Flashback's two yaml wrappers, with "anomalous event" as the action category | Flashback Fig. 2, p.4 |
| BD | B with the templates, each phrase's type as its action category | |
| TD | T with the templates, likewise | |

The twelve types are fighting, weapon, theft, vandalism, road accident, fire or explosion, riot or arrest,
abuse, falling, vehicle among pedestrians, sudden running and trespass. The bank is the same for every dataset.

Differences from the sources:
- NOVA writes 20 normal and 20 anomaly phrases per class, with a class list and hand-written guidance per
  dataset. Here there are 4 per type, one shared normal side and one list for all datasets. ASK-Hint found
  that too many prompts hurt, and NOVA's own banks are made for crime benchmarks only.
- No geometry-guided rewriting, visual anchor or motion gate.
- NOVA uses α = 0.9 and τ = 0.03. Here α stays 1 and λ stays 100, so nothing else changes.
- Flashback retrieves captions per segment from a memory with a video encoder. Here only its templates are
  used, with frames and prototypes. Its wrappers contain "Normal" and "Anomalous", so D overlaps with K, which failed step 1.

### Gates

For each dataset: C through the new scoring path gives identical margins; C matches the fixes run; the
videos and the old LLaVA scores match the ViT-L run (difference ≤ 1e-5).

### Decision rule

- B, T and D are each judged against C by the rule.
- B and T use the same bank, so if both pass, the larger mean gain goes on.
- A pair (BD or TD) counts only if its two parts pass alone and the pair also passes. A pair whose
  parts failed is reported but cannot be chosen, even if it looks best.
- The current best becomes whichever eligible condition has the largest mean gain. If nothing passes,
  PE-Core + C stays.
- **Step 4 runs only if B or T passes**, because it needs a useful bank. Otherwise it is skipped.
- Against the old LLaVA model, the same three words as step 1 apply.

### Predictions

| | Prediction | Why |
|---|---|---|
| J1 | At least one of B, T and D passes the rule against C | These ideas account for much of the papers' lead |
| J2 | T reaches 0.70 on XD-Violence | NOVA's class-name template with this scoring reaches 77.94 |
| J3 | T lowers ShanghaiTech | Our per-phrase scoring helped XD and hurt ShanghaiTech in the full stack |
| J4 | D raises UCF-Crime | Flashback's largest gain from templates was on UCF |
| J5 | Nothing reaches NOVA's UCF-Crime 89.86 or XD-Violence 84.82 | Our setup is not tailored to those datasets |

### Limitations

- An LLM (Claude) wrote the bank before scoring, knowing these benchmarks' anomaly classes and our earlier
  results. The vehicle and running types come close to ShanghaiTech's and NWPU's classes, and most others to UCF-Crime's
  and XD-Violence's. So the bank is not independent of these datasets. Only the sealed test can show whether it transfers.
- One wording, one size and one set of types. Other banks would give other numbers.
- T compares individual phrases while B compares prototypes, so their margins differ in scale, and λ
  is not retuned for T. That may limit T.

### Run

```bash
cd ~/dazvad/repo && git pull && source ~/dazvad/venv/bin/activate
python notebooks/ideas_sweep.py selftest
python notebooks/ideas_sweep.py step3 2>&1 | tee ~/dazvad/logs/ideas_step3.log
```

---

## Step 4: phrases chosen by scene (written 5 November 2026, before scoring)

### Question

The bank (B) leans towards crime, and NWPU Campus fell when it was added, though within noise. Can the
system see what kind of scene a camera shows, and use only the anomaly types that fit that scene? This idea
came from the project's author. It is retrieval, because nothing is generated at detection time.

### How it works

1. Each unit (camera, scene or video, as in earlier runs) takes its first video's five calm frames, the
   same ones the earlier retrieval used, chosen by the plain detector without labels.
2. Their mean image vector is compared with eight scene sentences (`ideas.SCENES`), by cosine similarity.
3. The closest scene gives a list of anomaly types, and the phrases of those types form the abnormal side.
   No new phrase is written, so each scene uses part of the frozen bank.
4. If no scene beats the neutral phrases ("a photo of a place" and two others), the self-check from the earlier
   retrieval fails, and the unit keeps the whole bank.

### Conditions (PE-Core, no sentence; normal side is always base + C)

| Name | Definition |
|---|---|
| B | Current best: the whole bank |
| S | The bank part for the closest scene |
| S_wrong | Control: the bank part for the least-matching scene. Units that fail the self-check keep the whole bank here too |
| C, old_llava | As before, for reference |

### Gates

For each dataset: B and C match step 3; the videos and the old LLaVA scores match the earlier runs
(difference ≤ 1e-5). A dataset that fails is not reported.

### Decision rule

S replaces B only if **both** hold:
- S passes the rule against B;
- S beats S_wrong with a CI above 0 on at least 2 datasets.

The second condition matters because without it, a gain could come from having fewer phrases rather than
from knowing the scene, and the extra step would not have earned its place. If either fails, S is dropped and B stays.

### Predictions

| | Prediction | Why |
|---|---|---|
| K1 | S fails the rule against B | Our earlier retrieval disappointed, and a smaller abnormal side can miss real anomalies |
| K2 | S is above B on NWPU Campus | Campus scenes lose the crime phrases that may hurt there |
| K3 | S beats S_wrong, CI above 0, on at least 2 datasets | The wrong scene removes the types that fit |
| K4 | XD-Violence falls back to the whole bank more often than any other dataset | Its films, games and sports fit no surveillance scene |
| K5 | S lowers XD-Violence | Its videos are the hardest to place in a scene |

### Limitations

- An LLM (Claude) wrote the scene sentences and the scene-to-type table before scoring, knowing these
  benchmarks. Other tables would give other numbers.
- The calm frames come from the test videos, so this is transductive. In deployment they would be setup footage,
  and frames already showing an incident could pick the wrong scene.
- A video's abnormal side can now differ from another's, which shifts score levels between videos. The audit reports this.
- One best scene is chosen, with no blending. A scene the table lacks gets its closest match, unless the self-check catches it.

### Run

```bash
cd ~/dazvad/repo && git pull && source ~/dazvad/venv/bin/activate
python notebooks/ideas_sweep.py selftest
python notebooks/ideas_sweep.py step4 2>&1 | tee ~/dazvad/logs/ideas_step4.log
```

---

## Step 5: camera memory (written 5 November 2026, before scoring)

### Question

Everything so far compares a frame with text. Does it help to also ask how unlike this camera's own past a
frame is? That needs no words and no training, and it directly tests the idea that normal depends on the camera.

### How it works

- **Memory:** each unit (camera, scene or video, as before) goes through its videos in name order and its frames
  in time order. A frame sees only earlier frames of its own unit, so no future frame is used.
- **Novelty:** 1 minus the mean cosine similarity to the 5 nearest earlier frames, leaving out the latest w sampled
  frames, where w is the dataset's existing smoothing window (31 on ShanghaiTech, 5 elsewhere). Those frames would match almost anything.
- **Combining:** the margin plus the novelty's deviation from its unit mean, scaled to the margin's spread within
  that unit. So the two count equally, and the margin's level across units is kept. No weight is tuned.
- A frame with fewer than 5 earlier frames has no novelty and keeps its margin.

### Conditions (PE-Core, no sentence; text is B throughout)

| Name | Definition |
|---|---|
| B | Current best |
| M | B with the unit's own memory |
| M_wrong | Control: B with the next unit's frames as memory, all of them, combined the same way |
| N_only | Novelty alone, with no text, point estimate only |
| C, old_llava | As before, for reference |

### Gates

For each dataset: B and C match step 3; the videos and the old LLaVA scores match the earlier runs
(difference ≤ 1e-5). A dataset that fails is not reported.

### Decision rule

M replaces B only if **both** hold:
- M passes the rule against B;
- M beats M_wrong with a CI above 0 on at least 2 datasets, so the gain comes from the unit's own past.

If either fails, M is dropped and B stays.

### Predictions

| | Prediction | Why |
|---|---|---|
| L1 | M fails the rule against B | Our simple visual anchor failed, and novelty also fires on harmless change |
| L2 | M raises NWPU Campus | Scene-dependent anomalies are what a camera's own past should catch |
| L3 | M lowers XD-Violence | Its videos contain edits and scene cuts, which look novel without being anomalies |
| L4 | M beats M_wrong, CI above 0, on at least 2 datasets | A camera's own past should be the better reference |
| L5 | N_only is below B on every dataset | Novelty alone knows nothing about what is dangerous |

### Limitations

- Memory comes from test videos, which contain anomalies, and a long incident soon starts to look familiar. In
  deployment it would start from setup footage, and the benchmarks' normal training videos are not used here.
- On UCF-Crime and XD-Violence each video is its own unit, so memory is short and starts empty.
- The scale and mean of the novelty use the whole unit, so this is not yet fully causal. A live system would need running estimates.
- Name order stands in for time order between a unit's videos, and those videos may not be consecutive.
- k = 5 and equal weighting are single choices, not tuned or tested.

### Run

```bash
cd ~/dazvad/repo && git pull && source ~/dazvad/venv/bin/activate
python notebooks/ideas_sweep.py selftest
python notebooks/ideas_sweep.py step5 2>&1 | tee ~/dazvad/logs/ideas_step5.log
```

## Addenda

Changes to a step after its commit go here, dated, and never in that step's text.

**5 November 2026, step 2 outcome.** All gates passed at commit e312e97. E and EC both fail the
rule against C: only XD-Violence gains significantly, and EC is significantly worse on UCF-Crime.
The idea is dropped, and the current best stays PE-Core + C. Predictions H2 and H3 held; H1 and H4 were refuted.

**5 November 2026, step 3 outcome.** All gates passed at commit f42b917. B passes the rule against C,
gaining significantly on UCF-Crime and XD-Violence, and is the current best. T fails (significantly worse on
ShanghaiTech), D fails (significantly worse on NWPU Campus), and the pairs cannot be chosen because their parts
did not both pass. All five predictions held. Step 4 runs, because the bank proved useful.

**5 November 2026, step 4 outcome.** All gates passed at commit a95eb27. S fails the rule against B, with all
four differences within noise, so it is dropped and B stays the current best. The wrong scene is significantly worse
than B on all four datasets, and S beats it significantly on XD-Violence and NWPU Campus. All five predictions held.

**6 November 2026, step 5 outcome.** All gates passed at commit e8de2f0. M fails the rule against B: it is
significantly worse on ShanghaiTech (−0.033) and XD-Violence (−0.117), and the other two differences are within
noise. Its own memory beats another unit's significantly only on UCF-Crime. M is dropped, and B stays the current
best. Predictions L1, L2, L3 and L5 held; L4 was refuted.

**6 November 2026, step 6 skipped.** The per-video baseline was not run. The reason is a judgement, not a
measurement: step 5 showed that adjusting each unit separately hurt, and a per-video baseline would remove the
differences in level between videos that B's gains on UCF-Crime and XD-Violence partly rest on. The round ends here
with PE-Core + C + B as the current best.

**5 November 2026, step 1 outcome.** All gates passed at commit 7168814. The second wording
passes the rule, so C counts as real. K fails: it gains significantly on three datasets, but its
CI on NWPU Campus is below 0 (−0.014 [−0.026, −0.003]). The current best is PE-Core + C.
Predictions G1, G4 and G5 held; G2 and G3 were refuted.
