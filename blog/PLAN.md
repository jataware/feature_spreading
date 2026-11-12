# Blog plan

Working title: **Fix the graph, not the classifier: few-shot labels for frozen vectors without training a network**

Status legend: `[ ]` todo · `[~]` running / partial · `[x]` done. This file is updated as work lands.

## 1. The claim, in one paragraph

You have a pile of frozen vectors and one to five labels per class. Without training any network:

1. smoothing over a *symmetric* k-nearest-neighbor graph with personalized PageRank (labels, vectors, or both) is far ahead
   of the semi-supervised methods in scikit-learn and at least level with the best in graphlearning (Poisson learning,
   random walk). How it compares with those two is for the dev table to settle, not this paragraph. And
2. the graph itself can be improved with **no labels**: k-means → ridge regression onto the cluster indicators → use `WᵀW`
   (the model's AGOP) as the metric for a second kNN graph.

Both halves are ours; the baselines are library implementations only.

On the six development datasets this took a linear probe from 0.62 to 0.72 (plain graph) to 0.75–0.78 (cluster-metric graph)
at 1-shot. The post has to show that this survives (a) vectors from other encoders and modalities, (b) honest cheap baselines,
(c) hyperparameters that were not tuned on the test data.

What is and is not new (to be checked against the papers before publishing):
- **Not new:** PPR / label spreading on a kNN graph (Zhou et al. 2003); smoothing features before a linear classifier
  (SGC, S²GC, APPNP for graphs; "embedding propagation" for few-shot images). The post should say so plainly.
- **Probably new, and the point of the post:** a label-free metric from clusters via ridge-AGOP to rebuild the graph;
  the K-must-be-large finding; the negative results (few-shot AGOP loop, MLP AGOP, self-training, edge pruning);
  the sklearn directed-graph pitfall.

## 2. Protocol (fixed before any held-out dataset is touched)

- **Task:** transductive few-shot classification. A dataset is a matrix of vectors; draw `n ≤ 5,000` points, label `s` per class,
  predict the rest. Accuracy on the unlabeled points.
- **Shots:** 1 and 5 in the main table; 1, 2, 5, 10, 20 in the shots curve.
- **Seeds:** 10 per (dataset, shots). Each seed redraws the sample and the labeled set. Report mean and a 95% interval;
  compare methods with paired differences, win/tie/loss over datasets, and average rank.
- **Dev / held-out split of datasets.** The six CLIP image datasets used so far are **dev**: every default below, and every
  baseline's hyperparameters, are chosen on them. Everything else is **held-out** and is run once with frozen settings.
  The post reports the two groups separately.
- **Preprocessing:** rows L2-normalized. Nothing else.
- **No modality-specific tricks.** No class names, no text prompts, no augmentation. Vectors and a few labels only.

### The recommended method, frozen

```
X  = normalize(X)
c  = KMeans(K=100).fit_predict(X)                       # no labels
W  = ridge(X - mean, onehot(c), penalty = 2e-4 * n)     # = 1 at n = 5,000, the tuned value
M  = (1 - λ) I + λ · WᵀW · d / tr(WᵀW),   λ = 0.5
A  = symmetric kNN graph on normalize(X M^½),  k = 10,  one self-loop
predict:  label spreading   argmax (I - αÂ)⁻¹ Y,  α = 0.95        ("CMG-LS")
     or:  LinearSVC on normalize((I - αÂ)⁻¹ X),   α = 0.85        ("CMG-FS")
```

Changes from the exploratory code, all checked in `experiments/`: fit on raw vectors instead of spread ones (same accuracy,
one graph fewer), a clean 10 neighbors instead of the accidental 8 (within 0.6 points), ridge penalty scaled with n.

## 3. Datasets

All end up as `data/<name>/<encoder>/{X,y}.npy` (at most 10,000 rows kept). Getting vectors is the only modality-specific step.

| group | name | source | encoder | classes |
|---|---|---|---|---|
| dev | pets, cars, resisc45, flowers, food101, fashion_mnist | (already on disk) | CLIP ViT-L/14 | 37–196, 10 |
| held-out text | newsgroups | `SetFit/20_newsgroups` | all-MiniLM-L6-v2 | 20 |
| | banking77 | `PolyAI/banking77` | all-MiniLM-L6-v2 | 77 |
| | clinc150 | `clinc/clinc_oos` (plus, out-of-scope dropped) | all-MiniLM-L6-v2 | 150 |
| | dbpedia14 | `fancyzhx/dbpedia_14` (test split) | all-MiniLM-L6-v2 | 14 |
| | agnews | `fancyzhx/ag_news` | all-MiniLM-L6-v2 | 4 |
| | emotion | `dair-ai/emotion` | all-MiniLM-L6-v2 | 6 |
| held-out audio | esc50 | `ashraq/esc50` | CLAP (htsat-unfused) | 50 |
| held-out image, other encoder | cifar100 | `uoft-cs/cifar100` | DINOv2 ViT-B/14 | 100 |
| | dtd | `tanganke/dtd` | DINOv2 ViT-B/14 | 47 |
| held-out, no neural encoder | mnist | OpenML `mnist_784` | raw pixels | 10 |
| | isolet | OpenML `isolet` (spoken letters, 617 acoustic features) | raw | 26 |
| | har | OpenML `har` (phone accelerometer, 561 features) | raw | 6 |

Encoder ablation (same data, different vectors): pets with DINOv2 and CLIP; newsgroups + banking77 with bge-base as well as MiniLM.

Left out on purpose: Speech Commands and TREC (script-only datasets, no longer loadable), UrbanSound8K (7 GB), CUB (1.1 GB; add if wanted).

- [x] `blog/data.py`: registry + featurizers (`python blog/data.py <name>`; skips what is on disk)
- [x] held-out vectors on disk: all 12

## 4. Baselines and our methods

**Baselines are methods as implemented in an existing library.** Nothing reimplemented by us counts as a baseline.

| library | method | notes |
|---|---|---|
| scikit-learn, labeled points only | `LinearSVC`, `LogisticRegression`, `NearestCentroid`, `KNeighborsClassifier(1)` | |
| scikit-learn, semi-supervised | `LabelPropagation`, `LabelSpreading` (rbf and knn kernels) | as shipped; the knn kernel builds a directed graph |
| | `SelfTrainingClassifier` around `LogisticRegression` | |
| | `KMeans` seeded with the labeled class means | cluster-then-label |
| graphlearning (Calder) | Laplace learning, Poisson learning, random walk, graph nearest neighbor, centered kernel | on the kNN graph that library builds (Gaussian weights) |

Left out: `poisson_mbo` (needs the true class proportions), `sparse_label_propagation` (7% on a first test, very slow),
anything that trains or fine-tunes a network, zero-shot text prompts.

**Ours** (`blog/ours.py`), all personalized-PageRank smoothing over a *symmetric* kNN graph:

| | plain kNN graph | cluster-metric graph |
|---|---|---|
| label spreading: spread one-hot labels, take the max | `ls` | `cmg_ls` |
| feature spreading: spread the vectors, linear SVM on labeled rows | `fs` | `cmg_fs` |
| both: add the two methods' standardized class scores | `fs+ls` | `cmg_fs+ls` |

The ensemble was the only combination that helped in a quick test (3 seeds, 6 datasets): it tracks the better of the two at each
shot count and adds about 0.3 points. Chaining them (spread the classifier's predictions afterwards) is worse than either.

Published methods we would have to reimplement (SGC, SimpleShot, soft k-means, LaplacianShot, TIM) are related work to
discuss, not baselines.

Each method has a small grid; the setting is chosen per shot count on the dev datasets (`blog/table.py`).

- [x] `blog/lib.py`: the method (graph, PPR, cluster metric) + shared runner
- [x] `blog/baselines.py`, `blog/ours.py`
- [x] `blog/run.sh`: data → baselines → ours → table, resumable
- [~] full run on the six dev datasets, 10 seeds (`blog/results/summary.md`)
- [ ] freeze the chosen settings, then run the held-out datasets once

## 5. Ablations and sensitivity (dev datasets; confirm the headline ones on held-out)

One factor at a time around the frozen defaults, both predictors, 1-shot and 5-shot. `blog/run_sens.py` covers every
single-hyperparameter item below plus rounds, k-means variance, metric type and targets (written; not yet run).

- [ ] clusters `K`: 10, 20, 50, 100, 200, 400, 800
- [ ] metric strength `λ`: 0, 0.1, 0.25, 0.5, 0.75, 1
- [ ] ridge penalty per sample: 2e-6 … 2e-2
- [ ] neighbors `k`: 5, 10, 15, 20, 30, 50
- [ ] restart `α`: features 0.5–0.95, labels 0.5–0.99
- [ ] rounds: 1, 2, 3
- [ ] k-means: variance over k-means seeds; MiniBatchKMeans
- [ ] what the metric is: ridge-AGOP vs shrinkage LDA on the same clusters vs PCA projection vs whitening vs random targets
- [ ] targets: clusters vs pseudo-labels vs both

Beyond single hyperparameters:

- [ ] **shots curve** 1 → 20: where does the gain over a linear probe go to zero?
- [ ] **n curve** 500 → 10,000 unlabeled points: how much data does the graph need?
- [ ] **encoder quality**: same dataset, weaker and stronger encoder
- [ ] **class imbalance**: long-tailed class sizes
- [ ] **runtime** vs n, with a sparse implementation
- [ ] **when does it help?** gain vs edge homophily of the plain graph, across all datasets

Already done in `experiments/` and only cited: kernel vs linear AGOP, MLP AGOP, few-shot AGOP loop, self-training,
edge pruning, sklearn label propagation diagnosis, retrieval and active retrieval (separate story).

The curves reuse `baselines.py` / `ours.py` (`--shots 1 2 5 10 20`, `--n 500 … 10000`, `--encoder …`).

Early signal worth a section of its own (one pets seed, to be confirmed): shrinkage **LDA fit to the k-means clusters gives the
same graph quality as ridge-AGOP**. If that holds, the honest one-line description of the metric is “regularized LDA on
cluster pseudo-labels”, and AGOP is the route that led there.

## 6. Deliverables

- [x] `blog/run.sh`: one command from nothing to the results table
- [x] `blog/lib.py`: the method in well under 100 lines
- [~] `blog/results/summary.md`: main table (dev now, held-out after freezing)
- [ ] sensitivity grid, shots curve, n curve, "when does it help" scatter, runtime table
- [ ] `blog/post.html`: the post, figures generated from the CSVs

## 7. Order of work

1. ~~`lib.py`, `data.py`, baselines, ours, `run.sh`~~ done
2. dev table (running) → freeze settings
3. held-out datasets, once
4. sensitivity and the curves
5. write the post
