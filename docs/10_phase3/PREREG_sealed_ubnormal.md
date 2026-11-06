# Freeze and sealed test: PE-Core + C + phrase bank on UBnormal

Written 6 November 2026, before any UBnormal video, frame or label has been downloaded or seen.
This note freezes the model chosen in `PREREG_ideas.md` and fixes, in advance, how it is tested once on
a dataset that played no part in choosing it.

## Why

Every number so far comes from the four datasets used to choose the model: ShanghaiTech, UCF-Crime,
XD-Violence and NWPU Campus. Many ideas were tried on them, and the phrase bank was written by an LLM that knew
their anomaly types. Those numbers therefore show how well the model fits data it was chosen on, not how it does
on new data. This test answers the second question, once.

## The frozen model

Nothing below may change after this note is committed.

| Part | Setting |
|---|---|
| Encoder | PE-Core-L-14-336 (open_clip, pretrained "meta"), frozen, whole frame, one crop |
| Normal side | The three "surveillance" phrases (`da_zvad/prompts/templates.py`) plus the eight calm phrases `fixes.CALM_NORMAL` |
| Abnormal side | The 48-phrase bank `ideas.BANK_PHRASES` |
| Prototypes | Each side's unit text embeddings averaged, then renormalised |
| Margin | ⟨f, abnormal prototype⟩ − ⟨f, normal prototype⟩, per sampled frame |
| Score | sigmoid(100 × margin), then a centred moving average over 5 sampled frames |
| Sampling | Every 16th decoded frame, starting at frame 0 |
| Normalisation | None (raw scores), as on UCF-Crime, XD-Violence and NWPU Campus |
| Sentence, retrieval, memory | None |

This is condition B of step 3, as computed by `notebooks/ideas_sweep.py` at commit 5a2dafb. File hashes at freezing
(sha256, first 16 characters, and the commit that last changed each file):

| File | sha256 | Commit |
|---|---|---|
| `da_zvad/prompts/ideas.py` | d889dbc6a76a4410 | a95eb27 |
| `da_zvad/prompts/fixes.py` | 32ab541b436c77f5 | 9ae7671 |
| `da_zvad/prompts/banks.py` | 0ab005d55f05a5a8 | a87ff87 |
| `da_zvad/prompts/templates.py` | 5a6456c8bf24bca5 | 2290df8 |
| `notebooks/ideas_sweep.py` | d0ee71aca2ad9c5b | e8de2f0 |
| `notebooks/fixes_sweep.py` | 5efe7532daafff5c | 9ae7671 |
| `notebooks/normal_side_sweep.py` | d41be068c7dd1b4b | a87ff87 |

The sealed-test script must reproduce condition B's scores on all four development datasets (difference ≤ 1e-5)
before it touches UBnormal. If any of these files changes, the test does not run.

The sampling step, smoothing and lack of normalisation follow three of the four development datasets.
ShanghaiTech's per-clip normalisation is that benchmark's own protocol and is not carried over.

## The dataset

UBnormal (Acsintoae et al., CVPR 2022): 29 virtual scenes rendered at 30 frames per second, 236,902 frames in total,
of which 92,640 are in the test split (Table 1, p.3). Its test anomaly types differ from its training types by design.
It is synthetic, which the results must state.

- **Split:** the official test split only, listed in `normal_test_video_names.txt` and `abnormal_test_video_names.txt`
  in the authors' repository (github.com/lilygeorgescu/UBnormal, folder `scripts`). The training and validation
  videos are not used at all, so the model stays training-free.
- **Labels:** frame-level labels exactly as the authors' `write_frame_level_ground_truth.py` produces them, ported
  literally and checked on synthetic input before use.
- **Download:** the authors' single Google Drive file. If Drive's download limit blocks it, we wait and retry.
  The limit is not worked around.
- **Licence:** CC BY-NC-ND 4.0. No frame, video or label goes into the repository; only file names, counts,
  checksums and our scores do.

## Order of work

1. This note is committed.
2. The sealed-test code is written and committed: the UBnormal loader, the label port, the scoring, and a
   self-test on synthetic data. It must pass the development gates above.
3. The dataset is downloaded on the server. Before scoring, only file counts and checksums are checked. No video is
   played, no frame is opened and no label is read by a person.
4. Test frames are encoded with PE-Core, and with ViT-L for the reference condition.
5. The scoring runs **once**. Its output is copied, committed and reported in a dated addendum here, whatever it shows.

If the run fails for a technical reason, such as a crash, a missing file or a failed gate, it may be fixed and rerun.
The fix and the reason go in the addendum. A rerun is never allowed because of the score.

## What is measured

- **Primary:** micro-averaged frame-level AUROC over all test frames, as in the authors' `compute_auc_score.py`
  (all frames pooled). Each sampled frame's smoothed score is held for the frames up to the next sampled frame, so
  every frame has a score.
- **Secondary:** macro-averaged frame-level AUROC (the mean over videos, with the authors' padding of one 0 and one 1
  per video), and the micro AUROC at sampled frames only, the measure used in our earlier runs.
- **Confidence intervals:** paired video-level bootstrap, 1000 resamples, seed 20261104, as before.
- **Reference conditions,** scored in the same run and never chosen between: plain PE-Core (no C, no bank),
  PE-Core + C, and plain ViT-L. The old LLaVA model is not run, because it would need new captions for every video.

## What each result allows us to say

Fixed now, so the wording cannot be chosen after seeing the numbers.

| Result | What we may say |
|---|---|
| B's 95% CI lies above 0.5 | "Detects anomalies on an unseen dataset, without training" |
| B's 95% CI includes 0.5 | "Did not detect anomalies reliably on UBnormal" |
| B − plain PE-Core, CI above 0 | "The calm phrases and bank carried over to unseen data" |
| B − plain PE-Core, CI includes 0 | "Unseen data did not show a gain from the calm phrases and bank" |
| B − plain PE-Core, CI below 0 | "The calm phrases and bank hurt on unseen data" |
| B against plain ViT-L | "better", "no worse" or "worse", by the same rule as `PREREG_ideas.md` step 1 |

**Published numbers:** UBnormal results elsewhere include AnomalyRuler (71.9, few normal frames and GPT-4), OVVAD
(62.94, trained) and the authors' baselines. A comparison may be stated only as "above" or "below" a published figure,
and only after that figure's metric (micro or macro, test split) has been checked in its PDF and entered in
`docs/07_reference_papers/PROVENANCE.md`. Words like "beats" or "state of the art" are not used.

## Predictions

| | Prediction | Why |
|---|---|---|
| U1 | B's 95% CI lies above 0.5 | The model detects anomalies on all four development datasets |
| U2 | B − plain PE-Core is above 0 | The calm phrases helped on three of four development datasets |
| U3 | B's micro AUROC is between 0.60 and 0.75 | Published UBnormal results we have seen lie between about 0.50 and 0.72. The data is synthetic, and its anomaly types include actions such as crawling, dancing and sleeping (p.2) that our bank does not describe |

## Limitations

- UBnormal is computer-generated, so a result here may not carry over to real cameras.
- Its test anomalies include types the bank does not name. That is part of the test, not a flaw in it.
- One dataset is one test. A good result here supports the model, and a poor one does not prove it useless.
- Choosing UBnormal was settled earlier in `PREREG_normal_side.md`. MSAD stays optional and would need its own note.

## Addenda

The result and any technical fix go here, dated.

**6 November 2026, before any UBnormal data was downloaded: a technical clarification about one hash.**
The hash table above was computed on a Windows working copy. There `da_zvad/prompts/templates.py` has
Windows line endings, which give `5a6456c8bf24bca5`; the committed file, the same content with Linux line
endings as the server checks it out, gives `4de350b2382843e4`. The file has not changed since commit 2290df8.
The sealed-test script therefore compares every frozen file's hash with line endings normalised to LF; the
other six hashes are identical under both. Nothing about the model or the test changes.

**6 November 2026, before any UBnormal data was downloaded: the sealed-test code.** Step 2 of the order of
work is `da_zvad/datasets/ubnormal.py` (loader, and a literal port of the authors' frame-level labels:
`gt[start:end + 1] = 1` per track, frame count from OpenCV) and `notebooks/sealed_ubnormal.py` (hash check,
development gates, the four conditions, the micro, macro and sampled metrics, paired video-level bootstrap,
and the wording and predictions of this note), with a self-test that passes on synthetic data. The script
refuses a second run unless given a technical reason, which it records. One consequence of the frozen rule is
made explicit rather than changed: the step-1 rule needs significant gains on at least two benchmarks, so on
one benchmark the comparison with plain ViT-L can only return "no worse" or "worse".

**6 November 2026: the sealed result.** The test ran once on the server at commit 5c52d41, with no
technical rerun. Before scoring, the seven frozen files matched their hashes, the frozen configuration
reproduced its step-3 scores on all four development datasets exactly (0.782956, 0.855806, 0.722330,
0.638771), all 158 abnormal and 53 normal test videos and every tracks file were present, and the sampled
labels in the embedding cache agreed with the frame-level labels. The run covered 92,640 test
frames (49,850 anomalous), the figure this note took from the paper.

| Condition | Micro AUROC (primary) | 95% CI | Macro AUROC | Micro, sampled frames |
|---|---|---|---|---|
| B: PE-Core + C + bank (frozen model) | 0.7475 | [0.7044, 0.7886] | 0.9063 | 0.7485 |
| PE-Core + C | 0.6874 | [0.6344, 0.7414] | 0.8962 | 0.6900 |
| Plain PE-Core | 0.6512 | [0.5988, 0.7084] | 0.8580 | 0.6538 |
| Plain ViT-L | 0.5865 | [0.5223, 0.6443] | 0.8129 | 0.5883 |

Paired video-level bootstrap, 1000 resamples of 211 videos, seed 20261104:
B − plain PE-Core +0.0963 [+0.0509, +0.1424];
B − plain ViT-L +0.1610 [+0.1117, +0.2150];
B − C +0.0601 [+0.0158, +0.1024];
C − plain PE-Core +0.0362 [+0.0057, +0.0690].

**What the result allows us to say,** using only the wording fixed above:
"Detects anomalies on an unseen dataset, without training" (B's interval lies above 0.5);
"The calm phrases and bank carried over to unseen data" (B − plain PE-Core, interval above 0);
against plain ViT-L, "no worse" (the step-1 rule needs two benchmarks to return "better"; it is applied
literally, although the interval of B − plain ViT-L lies well above 0).

**Predictions:** U1 holds; U2 holds; U3 holds, narrowly (0.7475 against an upper bound of 0.75).

**Published figures,** checked in their PDFs and entered in `docs/07_reference_papers/PROVENANCE.md`
before being stated here. All are micro frame-level AUROC on the test split:
B's 74.7 is above the UBnormal authors' baselines (best 68.5, a supervised model fine-tuned on
UBnormal; Georgescu et al. 59.3, and 61.3 with UBnormal's training anomalies), above OVVAD (62.94, trained),
FPDM (62.7) and STG-NF (71.8), and above AnomalyRuler (71.9, a few normal frames and GPT-4). AnomalyRuler's
figure lies inside B's 95% interval, so that difference is not established. On macro AUROC B's
90.6 is above every macro figure in the authors' Table 2 (best 85.6).

**Limits that apply to this result,** as stated before it was known: UBnormal is computer-generated; it is
one dataset; and the micro AUROC of all four conditions is computed with each sampled frame's score held
for the next fifteen frames, as fixed above.
