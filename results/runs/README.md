# Experiment runs

Measured results from GPU runs, kept under version control. Everything else
under `results/` is gitignored scratch output from synthetic checks; this folder
is the permanent record.

The runs themselves were executed on a shared institutional machine that gets
reimaged and whose accounts expire, so the server is not the archive — these
files are.

## Layout

```
<UTC date>_<time>_<tag>/
    MANIFEST.txt        conditions the run was executed under
    tables/
        grid.csv            dataset x context on/off x temporal window
        context_sweep.csv   none / generic / matched / mismatched
        reanalysis.csv      pooling protocols recomputed from cached scores
scoring_lab/
    scoring_lab.csv     scoring strategies, single dev/held-out partition
    scoring_lab2.csv    with motion scoring, five partitions
```

Each `MANIFEST.txt` records the git commit and whether the working tree was
clean, the host, GPU, driver and library versions, the full configuration, and
per-dataset sequence/frame/positive counts — so any figure in the paper can be
traced to the code and hardware that produced it, and the run can be repeated.

## The runs

| Run | Configuration | Outcome |
|---|---|---|
| `131113_shanghaitech` | surveillance prompts, context fused into **both** ensembles | Sweep inverted its own prediction: matched worst (0.637), mismatched best (0.678) |
| `145146_campus_normal` | campus prompts, context into **normal** ensemble only | Campus anomaly vocabulary performed far worse in isolation (0.486) |
| `162056_surv_normal` | surveillance prompts, context into **normal** ensemble only | Predicted signature: matched 0.692, none 0.685, mismatched 0.592 |

The three differ by one variable at a time. `none` is identical across the first
and third (0.685), which confirms nothing but the fusion rule changed between
them.

AUROC figures above are per-clip normalised micro at window 5. Raw and macro
figures are in the CSVs; the paper reports all three, because raw pooling on
this benchmark sits near chance and reporting only the favourable convention
would misrepresent the result.

## Reproducing

```bash
python notebooks/run_experiments_server.py \
    --shanghaitech <root> --domain surveillance --context-mode normal \
    --frame-step 2 --tag surv_normal
```

The scoring lab needs a cached embedding pass first
(`notebooks/cache_embeddings.py`), after which it re-evaluates scoring
hypotheses without touching the GPU.
