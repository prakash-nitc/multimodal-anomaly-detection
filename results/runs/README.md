# Experiment runs

Measured results from GPU runs, kept under version control. Everything else
under `results/` is gitignored scratch output from synthetic checks; this folder
is the permanent record.

The runs were executed on a shared institutional machine that gets reimaged and
whose accounts expire, so the server is not the archive — these files are.

## Layout

```
<UTC date>_<time>_<tag>/
    MANIFEST.txt        conditions the run was executed under
    tables/             that run's own output
analysis/               post-hoc analysis over cached scores and embeddings
```

Each `MANIFEST.txt` records the git commit and whether the working tree was
clean, the host, GPU, driver and library versions, the full configuration, and
per-dataset sequence/frame/positive counts — so any figure in the paper can be
traced to the code and hardware that produced it, and the run can be repeated.

## The runs

| Run | Configuration | Outcome |
|---|---|---|
| `08-14_131113_shanghaitech` | surveillance prompts, context into **both** ensembles | Sweep inverted its own prediction: matched worst, mismatched among the best |
| `08-14_145146_campus_normal` | campus prompts, context into **normal** only | Campus anomaly vocabulary far worse in isolation (0.486) |
| `08-14_162056_surv_normal` | surveillance prompts, context into **normal** only | Predicted signature: matched 0.734, none 0.707, mismatched 0.628 at w=31 |
| `08-18_133403_avenue` | Avenue, first attempt | Scene descriptor was factually wrong (said "subway station"); superseded |
| `08-18_140941_avenue_fixed` | Avenue, corrected descriptor | Detection transfers (0.706); context effect nearly absent (+0.020) |

The first and third differ by one variable. Their `none` columns agree to four
decimals, which they must if nothing else changed — that is the internal control,
and it is checkable here rather than merely asserted.

## Which analysis files are current

`analysis/` accumulated during a debugging session and **not all of it is
valid**. Superseded files are kept because the paper's limitation about the
metric's scale-sensitivity is evidenced by the discrepancy between them.

| File | Status |
|---|---|
| `within_view.csv` | **current** — computed from the driver's cached scores |
| `components_fixed.csv` | **current** — whole-frame, windows to 31 |
| `sweep_whole2.csv` | **current** — reproduces the driver's gap (+0.102 vs +0.105) |
| `sweep_whole.csv` | **superseded** — see below |
| `sweep_by_strategy.csv` | superseded — aggregates over five crops, so not comparable to the driver |
| `scoring_lab.csv`, `scoring_lab2.csv`, `scoring_lab3.csv` | superseded — same crop issue |
| `grid.csv`, `context_sweep.csv` | last driver run's tables, duplicated from `work/` |

Two defects produced the superseded files, both worth knowing about.

**Crop aggregation.** The scoring lab averaged over five crops while the
experiment driver scores whole frames. Every lab figure was therefore
incomparable to a driver figure until `--crop-agg whole` was added in `2b74a59`.

**The score transform.** The driver scores the softmax probability of the
abnormal prototype, `sigmoid(logit_scale * margin)`; the lab scored the margin
itself. That is a monotone transform, so within-clip AUROC is identical and a
rank correlation between the two score arrays comes back at exactly 1.0 — which
is why the defect survived a correlation check. Pooled AUROC is not identical,
because per-clip min-max normalisation is affine and an affine map applied after
a nonlinear monotone one is not the affine map alone. The two normalise each
clip onto [0, 1] differently and rank differently once clips are pooled.
Corrected in `3635fd4`.

`sweep_whole.csv` and `sweep_whole2.csv` were produced by the identical command
either side of that fix, and report gaps of +0.020 and +0.102 respectively. The
second reproduces the driver; the first does not.

The lesson generalises past this repository: **a rank correlation of 1.0 does
not establish that two scorings are equivalent under a metric that normalises
per clip.** Check the metric, not the ordering.

## Reproducing

```bash
python notebooks/run_experiments_server.py \
    --shanghaitech <root> --domain surveillance --context-mode normal \
    --frame-step 2 --tag surv_normal

python notebooks/within_view_control.py --raw <work>/raw --window 31
```

The scoring lab needs a cached embedding pass first
(`notebooks/cache_embeddings.py`), after which it re-evaluates scoring
hypotheses without touching the GPU. Pass `--crop-agg whole` for figures that
are comparable to a driver run.
