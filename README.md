# feature_spreading

Classify a pile of fixed vectors from a handful of labels, in seconds, without training a network.

```python
from feature_spreading import SpreadingClassifier

# X: (n, d) vectors, e.g. embeddings.  y: class labels, with -1 for every unlabeled row.
clf = SpreadingClassifier().fit(X, y)
clf.transduction_          # a predicted label for every row
```

## Why

A common situation: you have embeddings for a few thousand items (from an API, a frozen encoder, or a feature
pipeline you can't change), you can afford to label one to five examples per class, and you want labels for the rest
now. You don't want to fine-tune anything, and you may want to repeat this many times: inside an annotation loop,
for every new taxonomy somebody proposes, or as a quick check of whether an embedding model is good enough for a task.

A linear probe on five labels throws away what the unlabeled points tell you about where the data is dense and which
points belong together. Semi-supervised methods use that, but the ones that ship in common libraries each turned out to
have a setting where they are weak or break (see [Results](#results)). This library is the recipe that held up across
34 dataset and encoder combinations: images, text, audio and plain feature vectors.

- **Fast.** About three seconds for 5,000 vectors on a laptop CPU. Numpy, scipy and scikit-learn only.
- **Robust.** One default configuration, no per-dataset tuning, near the top on almost every dataset we tried.
- **Familiar.** A scikit-learn estimator with the same conventions as `sklearn.semi_supervised.LabelSpreading`.

## Install

```bash
pip install git+https://github.com/jataware/feature_spreading
```

## Use

```python
import numpy as np
from feature_spreading import SpreadingClassifier, SpreadingTransformer

y = np.full(len(X), -1)            # -1 = unlabeled
y[labeled_rows] = labels           # any label type works: ints, strings, ...

clf = SpreadingClassifier(random_state=0).fit(X, y)
clf.transduction_                  # predicted label for every row passed to fit
clf.scores_                        # (n, n_classes) scores behind them
clf.predict(X_new)                 # rows that were not in fit (approximate, see below)
```

The options that matter:

| parameter | default | what it does |
|---|---|---|
| `predictor` | `"both"` | `"vectors"`: smooth the vectors over the graph, then fit a linear SVM on the labeled rows. `"labels"`: smooth the labels over the graph (label spreading). `"both"`: add the two sets of class scores. |
| `metric` | `"cluster+pseudo"` | `None`: plain nearest-neighbor graph. `"cluster"`: rebuild the graph under a metric learned from k-means clusters; uses no labels. `"cluster+pseudo"`: also fit that metric to a first-pass guess of every row's class; uses the labels to build the graph. |
| `n_neighbors` | `10` | neighbors per row in the graph |
| `alpha` / `alpha_labels` | `0.85` / `0.95` | how far the smoothing reaches, for vectors and for labels |
| `n_clusters` | `100` | k-means clusters for the metric. Use at least about 100; it does not need to match the number of classes. |
| `metric_strength` | `0.5` | 0 = plain cosine similarity, 1 = the learned metric alone |
| `estimator` | `LinearSVC()` | any scikit-learn classifier, fit on the labeled rows of the smoothed vectors |
| `solver` | `"auto"` | `"dense"` (exact) up to 6,000 rows, `"iterative"` (sparse) beyond |

If you want the smoothed vectors themselves, for your own classifier, for retrieval, or to reuse one graph across many
label sets, use the transformer. It never sees a label:

```python
P = SpreadingTransformer(metric="cluster").fit_transform(X)     # (n, d), unit-norm rows
```

The pieces are also available as functions: `knn_graph`, `ppr_smooth`, `cluster_metric`.

## How it works

1. **Graph.** Link every row to its 10 most cosine-similar rows, and make the links symmetric (two rows are linked if
   either lists the other).
2. **Smooth.** Replace each row of a matrix `B` by a personalized-PageRank-weighted average over the graph:
   `(1 - α)(I - αÂ)⁻¹ B`. With `B` = the one-hot labels, this is label spreading. With `B` = the vectors themselves,
   you get smoothed vectors on which a linear classifier trained on five labels does much better.
3. **Better graph (optional).** Run k-means with many clusters, fit a ridge regression that predicts each row's cluster
   from its vector, and use `WᵀW` of that regression as a metric: directions that separate clusters get stretched.
   Rebuild the graph in that space. With `"cluster+pseudo"`, the regression also predicts a first-pass guess of each
   row's class, which points the metric at the classes you actually care about.

Nothing here trains a network, and the only iterative step is k-means.

## Results

Protocol: take up to 5,000 vectors from a dataset, label 1 or 5 per class, predict the rest, and measure accuracy on the
unlabeled rows. Ten seeds, each with a fresh sample and fresh labeled rows. Every method sees the same draws. Baselines are
called from scikit-learn and from [`graphlearning`](https://github.com/jwcalder/GraphLearning).

<!-- results:start -->
<!-- generated by blog/readme_assets.py from blog/results/ : do not edit by hand -->

Accuracy (%) on the unlabeled points, averaged over the datasets in each group and over 10 seeds.
Groups are named by what produced the vectors: CLIP (images, 6), DINOv2 (images, 6), Qwen3 (text, 6), Gemini (text, 6), MiniLM (text, 6), audio, raw (4). **Mean rank** is over all 34 dataset / encoder combinations,
among the 23 methods compared (1 is best). Bold marks the best entry in a column.
Rows for this library use its default settings throughout; each baseline gets the best setting from a small grid, chosen per shot count on the CLIP datasets.

**1 labeled example per class**

| | CLIP | DINOv2 | Qwen3 | Gemini | MiniLM | audio, raw | mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| **scikit-learn** | | | | | | | |
| `LinearSVC` | 61.4 | 60.1 | 51.4 | 59.4 | 45.8 | 60.2 | 15.9 |
| `SelfTrainingClassifier` | 53.9 | 55.3 | 53.6 | 59.8 | 48.8 | 57.0 | 16.7 |
| `LabelSpreading` | 62.3 | 63.4 | 57.8 | 65.6 | 51.8 | 65.9 | 13.0 |
| `KMeans`, seeded with the class means | 67.1 | 62.7 | 64.2 | **73.3** | 57.1 | 62.2 | 10.1 |
| **graphlearning** | | | | | | | |
| random walk | 73.1 | 66.1 | 55.9 | 60.3 | 50.6 | 72.8 | 9.2 |
| Poisson learning (graph made connected) | 72.8 | 65.6 | 58.3 | 61.6 | 51.8 | **73.6** | 8.9 |
| **this library, `predictor="both"`** | | | | | | | |
| `metric=None` | 72.6 | 65.9 | 63.9 | 71.2 | 58.0 | 71.8 | 7.2 |
| `metric="cluster"` | **77.6** | 65.4 | 63.7 | 68.4 | 59.4 | 72.9 | 7.1 |
| `metric="cluster+pseudo"` (the default) | 77.3 | **66.4** | **66.0** | **73.3** | **60.3** | 71.8 | **3.7** |
| **this library, `predictor="vectors"`** | | | | | | | |
| `metric=None` | 70.9 | 64.9 | 63.8 | 71.7 | 58.0 | 68.6 | 8.3 |
| `metric="cluster"` | 75.6 | 64.9 | 63.9 | 71.3 | 59.1 | 69.7 | 7.9 |
| `metric="cluster+pseudo"` | 75.6 | 65.6 | 65.6 | 73.2 | 59.8 | 69.2 | 5.7 |
| **this library, `predictor="labels"`** | | | | | | | |
| `metric=None` | 72.6 | 65.8 | 62.6 | 69.8 | 57.0 | 72.2 | 8.9 |
| `metric="cluster"` | **77.6** | 65.0 | 62.4 | 65.5 | 58.8 | 73.1 | 9.1 |
| `metric="cluster+pseudo"` | 77.4 | **66.4** | 65.9 | 72.7 | 60.2 | 72.0 | 4.5 |

**5 labeled examples per class**

| | CLIP | DINOv2 | Qwen3 | Gemini | MiniLM | audio, raw | mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| **scikit-learn** | | | | | | | |
| `LinearSVC` | 83.9 | **78.5** | 71.8 | 78.3 | 66.0 | 79.6 | 11.5 |
| `SelfTrainingClassifier` | 83.0 | 78.3 | 75.8 | 80.5 | 70.3 | 79.5 | 8.7 |
| `LabelSpreading` | 79.9 | 73.6 | 67.8 | 75.3 | 62.6 | 78.9 | 16.3 |
| `KMeans`, seeded with the class means | 79.8 | 71.6 | 74.7 | 80.6 | 68.6 | 68.3 | 11.9 |
| **graphlearning** | | | | | | | |
| random walk | 83.4 | 76.3 | 64.5 | 67.6 | 60.1 | 83.3 | 10.8 |
| Poisson learning (graph made connected) | 81.3 | 73.5 | 65.5 | 67.6 | 59.6 | 80.8 | 15.1 |
| **this library, `predictor="both"`** | | | | | | | |
| `metric=None` | 82.9 | 75.7 | 75.3 | 80.8 | 70.0 | 83.3 | 8.6 |
| `metric="cluster"` | 85.4 | 75.0 | 74.3 | 79.2 | 70.6 | 84.7 | 8.6 |
| `metric="cluster+pseudo"` (the default) | 86.4 | 76.9 | 78.2 | 83.4 | 72.7 | 85.7 | **3.3** |
| **this library, `predictor="vectors"`** | | | | | | | |
| `metric=None` | 82.4 | 76.1 | 76.9 | 82.5 | 71.7 | 82.2 | 7.6 |
| `metric="cluster"` | 84.8 | 75.8 | 76.3 | 82.4 | 71.8 | 83.9 | 7.2 |
| `metric="cluster+pseudo"` | 85.4 | 76.8 | **78.3** | **83.8** | **72.8** | 83.9 | 3.9 |
| **this library, `predictor="labels"`** | | | | | | | |
| `metric=None` | 82.6 | 75.0 | 73.0 | 78.6 | 67.9 | 83.0 | 11.6 |
| `metric="cluster"` | 85.3 | 74.4 | 72.1 | 76.5 | 69.0 | 84.3 | 11.4 |
| `metric="cluster+pseudo"` | **86.4** | 76.7 | 77.9 | 82.8 | 72.4 | **85.9** | 4.6 |

![Rank of each method on every dataset](assets/ranks.svg)

- **1-shot:** the default is among the top 5 of 23 methods on 28 of 34 datasets; the most consistent baseline (Poisson learning (graph made connected)) manages 11. Against the best baseline chosen separately for every dataset with hindsight, the default is +0.1 points on average and ahead on 16 of 34. It is more than 5 points behind on: mnist (-15.4, behind poisson).
- **5-shot:** the default is among the top 5 of 23 methods on 30 of 34 datasets; the most consistent baseline (KMeans, seeded with the class means) manages 10. Against the best baseline chosen separately for every dataset with hindsight, the default is +0.0 points on average and ahead on 19 of 34. It is more than 5 points behind on: cars_dinov2 (-22.6, behind LinearSVC).

<!-- results:end -->

The full comparison, with all 23 methods on every dataset, is in [`blog/results/summary.html`](blog/results/summary.html).

### What to take from this

- **The default is the most dependable method we tested**, not the best on every dataset. Different baselines win in
  different places: random walk and Poisson learning on images and raw features, seeded k-means on text with one label,
  self-training or a plain SVM with five. The default is near the top throughout.
- **`predictor="vectors"` is the safer half at 5 labels per class, `"labels"` at 1.** Adding them hedges.
- **The embedding model matters far more than the method.** Moving text from MiniLM to Gemini embeddings adds 12 to 16
  points for nearly every method in the table.

### Where it does worse

- **When the neighbor graph is poor, a linear probe can win by a lot.** The clearest case is Stanford Cars with DINOv2
  embeddings at 5 labels per class, listed above. If you have enough labels to hold some out, compare against `LinearSVC`.
- **MNIST pixels with one label per class**, where Poisson learning is far ahead.
- **The label-free metric (`metric="cluster"`) was tuned on CLIP image embeddings and mostly helps there.** The figure
  shows the gain from the metric as its strength goes up: large for CLIP, and negative for DINOv2 and Qwen3 beyond
  about 0.25. Adding pseudo-labels (`"cluster+pseudo"`, the default) is what made the metric help on every encoder.

![Gain from the label-free cluster metric by metric strength](assets/metric_strength.svg)

### Things we learned about the baselines

These are worth knowing even if you never use this library.

- **`sklearn.semi_supervised.LabelSpreading(kernel="knn")` builds a directed graph.** A labeled row only passes its label
  to rows that list it as a neighbor, which fails badly with one label per class. Symmetrizing the graph, with nothing
  else changed, brought it level with our label spreading (56.3% to 73.1% at one label per class, over 18 runs).
- **`graphlearning`'s Poisson learning collapses if the graph is not connected**, even when the second piece is twenty
  rows out of 5,000. The table uses a version where we raise the neighbor count until the graph is connected.
- **`graphlearning`'s default graph weights turn into NaN when rows are exact duplicates**, which broke its random walk
  on a text dataset containing empty documents.

## Speed and scale

Time for `fit` with `predictor="both"` on a laptop (Apple M-series, 8 threads), 768-dimensional vectors:

| rows | `metric=None` | `metric="cluster"` | `metric="cluster+pseudo"` (default) |
|---:|---:|---:|---:|
| 2,000 | 0.2 s | 0.3 s | 0.5 s |
| 5,000 | 1.4 s | 1.6 s | 2.8 s |
| 10,000 | 9 s | 9 s | 18 s |
| 50,000 | 99 s | 113 s | 211 s |

The first three rows use real embeddings; the 50,000-row line uses synthetic vectors of the same shape. Up to 6,000 rows
the smoothing is an exact dense solve; beyond that it is a sparse fixed-point iteration, which gives the same predictions.
The neighbor search is brute force, so time grows with the square of the number of rows: fine into the tens of thousands,
and you will want an approximate neighbor index beyond that. `"cluster+pseudo"` costs about twice the others because it
builds two graphs.

**Rows that were not in `fit`.** The method is transductive: its predictions are for the rows you fit on. `predict(X_new)`
places a new row by its nearest fitted rows, which is an approximation. In a test on four datasets it was within about
two points of refitting with the new rows included.

## What is new here, and what is not

Mostly this is a careful assembly of known ideas. A literature check is in [`blog/LITERATURE.md`](blog/LITERATURE.md).

- **Label spreading** is Zhou et al. (2003). Several recent papers use it on frozen embeddings.
- **Smoothing features over a graph before a simple classifier** is the idea behind SGC, and most directly GLP
  (Li et al., CVPR 2019) and Embedding Propagation (Rodríguez et al., ECCV 2020).
- **A metric from k-means labels** descends from LDA-Km and DIFFRAC (2007). **Rebuilding the graph under a metric learned
  from propagated labels** is IDML (Dhillon et al., 2010). **Gradient outer products as a metric** (the route by which we
  arrived at `WᵀW`) is Trivedi et al. (2014) and recursive feature machines (Radhakrishnan et al., 2024).
- What we did not find published: the label-free chain (over-clustered k-means → ridge metric → rebuilt graph → spreading),
  and a comparison of these cheap methods across encoders and modalities.

## Caveats

- Every setting was chosen on six CLIP image datasets. The other 28 dataset and encoder combinations were not used to choose
  the defaults, but two groups of them (DINOv2 images, Qwen3 text) were used to study sensitivity, which is how we
  arrived at `"cluster+pseudo"`. The MiniLM, Gemini, audio and raw-feature groups were only ever run once.
- Each run uses at most 5,000 rows. Behavior at much larger sizes is untested.
- Class sizes in these datasets are mostly balanced. Long-tailed data has not been tested.

## Reproducing the experiments

```bash
pixi install
pixi run ./blog/run.sh            # data -> baselines -> our methods -> blog/results/summary.html
pixi run python blog/readme_assets.py     # regenerate the tables and figures in this README
pixi run python tests/test_spreading.py   # or: pytest tests/
```

`blog/` holds the benchmark behind this README. `experiments/` holds the earlier exploration, including one-class
retrieval and an active-retrieval loop, with its own write-up in `experiments/report.html`.
