# experiments

One-off exploration code, kept so the results in `report.html` can be traced and re-run.
Nothing here is imported by the `fs` CLI.

**Read `report.html` first** (open it in a browser; it is a single standalone file). It explains the
methods from scratch and holds every result.

All scripts run from the repo root, read `data/<dataset>/<model>/{X,y}.npy` (created by `fs download`),
and write one CSV. None of them download anything.

| script | what it does | args |
|---|---|---|
| `01_agop_sweep.py` | Grid over the original `fs agop` loop (AGOP from the few-shot predictor) + the two oracle ceilings. Stopped early: pets, one seed. | `<dataset> <out.csv> [npc ...]` |
| `02_explore.py` | Stage 1: wide search over ways to combine SVM / PPR / AGOP (what to propagate, unsupervised graph tweaks, self-training, pseudo-label metrics, pruning, k-means metric). Pets, 5 seeds. | `<dataset> <npc> <seed> <out.csv> [full\|core]` |
| `03_explore2.py` | Stage 2: the two variants that worked (pseudo-label and k-means targets → ridge AGOP → graph) on five datasets, with first controls. Here K was tied to the class count. | `<dataset> <npc> <seed> <out.csv>` |
| `04_ablate.py` | Stage 3: tuning + ablations with K independent of the class count. Source of the classification sections of the report. Runs 1-shot and 5-shot together. | `<dataset> <seed> <out.csv>` |
| `05_retrieval.py` | One-class retrieval: 1 or 5 exemplars of a single class, rank everything else. | `<dataset> <seed> <out.csv>` |
| `06_sklearn_lp.py` | Why sklearn's kNN label propagation fails here (directed graph) and what fixes it. | `<dataset> <seed> <out.csv>` |
| `07_mlp_agop.py` | Swaps the ridge model for a one-hidden-layer MLP (closed-form AGOP). Worse than ridge with cluster / pseudo-label targets. Not in the report. | `<dataset> <seed> <out.csv>` |
| `08_active_retrieval.py` | Active retrieval: retrieve top 10, get noisy relevance labels, update the query model, repeat. Not yet in the report. | `<dataset> <seed> <out.csv>` |
| `09_knn_variants.py` | The repo's `k=10` graph is really 8 neighbors + a double self-loop; compares it with a clean 10. | `<dataset> <seed> <out.csv>` |
| `make_report.py` | Rebuilds `report.html` from `results/`. Uses `report_lib.py` (inline-SVG helpers). | none |

## Re-running

```bash
# e.g. all of stage 3, five at a time
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
for ds in pets cars resisc45 flowers food101 fashion_mnist; do for seed in 0 1 2; do
    echo "$ds $seed experiments/results/04_ablate/${ds}_${seed}.csv"
done; done | xargs -P 5 -L 1 pixi run python experiments/04_ablate.py

pixi run python experiments/make_report.py
```

Rough cost per (dataset, seed) on an M-series laptop with 3 threads: `04_ablate` 3–15 min,
`05_retrieval` 5–15 min, `06_sklearn_lp` under 2 min. Everything is dense linear algebra on n ≤ 5,000 points.

## Conventions in the CSVs

- `npc`: labeled examples per class (1 or 5). `seed` controls both the 5,000-point sample and the labeled set.
- `pred`: `svc` = LinearSVC on the labeled rows of the spread features; `lp` = PPR on one-hot labels over the same graph.
- `homophily`, and any method starting with `oracle_`, use **all** labels. They are diagnostics and ceilings, never methods.
- `results/01…03` use older column layouts (see each script's docstring); the report only quotes a few numbers from them.
