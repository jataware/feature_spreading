# Literature check: is any of this new?

Compiled 2026-11-10 from four parallel literature reviews (web search + reading abstracts / papers). Each review was told
to give a URL for every paper and to flag what it could not verify. **Nothing here has been independently re-read by a
person.** Items marked *(abstract only)* or *(unverified)* should be checked before they are cited in the post.

## Verdicts

| Piece of our method | Verdict | Confidence | Closest prior work |
|---|---|---|---|
| Spread the **labels** over a symmetrized kNN graph, closed form | Well known | high | Zhou et al. 2003; iLPC 2021 and ZLaP 2024 do exactly this on frozen embeddings |
| Spread the **vectors**, then a simple classifier on the labeled rows | Known under other names | high for the operation, medium for "nobody did this exact setting" | GLP (CVPR 2019); Embedding Propagation (ECCV 2020); SGC |
| Add the two predictors' standardized scores | Not found, but an incremental ensemble | medium-low | Embedding Propagation chains the two instead |
| **Cluster metric**: k-means (many clusters) → ridge → `WᵀW` metric → rebuild the kNN graph, no labels | Every piece exists; the chain was not found | medium | LDA-Km 2007, DIFFRAC 2007 (the metric); IDML 2010 (metric → rebuild graph, with labels); Trivedi et al. 2014 (gradient outer product as a kNN metric) |
| Pseudo-label variant (iterate: predict → metric → graph) | Close to already published | medium-high | IDML 2010 |
| Over-clustering (K ≫ classes) to learn a *linear* metric | Not found | low-medium (absence of evidence) | DeepCluster, IIC use over-clustering, but to train networks |

**What the post can honestly claim**

- Not a new propagation method. Label spreading is 2003; smoothing features before a classifier is GLP / SGC / Embedding Propagation.
- A careful, modality-agnostic comparison showing that this simple recipe is robust where each library method has a failure case.
- A label-free "discriminative-clustering metric" (LDA-like, on over-clustered k-means pseudo-labels) used to rebuild the graph,
  which none of the four reviews found published as such, together with the finding about where it helps.
- The scikit-learn directed-graph trap, demonstrated empirically (the asymmetry itself is a known open issue).

**What would be embarrassing to claim**

- That feature spreading is new (GLP is a near-identical recipe).
- That the pseudo-label loop is new (IDML).
- That label spreading beats Poisson learning in general: Calder et al. report the opposite on MNIST / FashionMNIST (see below).
- "Mutual kNN" for our graph: in the kNN literature *mutual* means intersection. Ours is the **union** (linked if either lists the other). Say "symmetrized".
- "No training": GLP and SGC also train only a shallow classifier on smoothed frozen features.

## 1. Spreading labels and vectors over a graph

- **Zhou, Bousquet, Lal, Weston, Schölkopf**, "Learning with Local and Global Consistency", NIPS 2003. https://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency . Origin of our label-spreading operator. Also "Ranking on Data Manifolds", NIPS 2003, https://papers.nips.cc/paper/2447-ranking-on-data-manifolds (the retrieval version).
- **GLP**: Li, Wu, Liu, Zhang, Guan, "Label Efficient Semi-Supervised Learning via Graph Filtering", CVPR 2019. https://arxiv.org/abs/1901.09993 *(full text read)*. Filters the features with `(I + αL)⁻¹`, which is our operator up to reparametrization, then trains any classifier on the labeled rows; notes that label propagation is the same filter applied to `Y`. Differences: given citation / knowledge graphs, an MLP, 5–20 labels per class. **The strongest evidence that feature spreading is not new.**
- **Embedding Propagation**: Rodríguez, Laradji, Drouin, Lacoste, ECCV 2020. https://arxiv.org/abs/2003.04151 *(full text read)*. Applies `(I − αL)⁻¹` to embeddings for 1–5-shot learning. Differences: dense Gaussian graph per episode, the backbone is trained with it, label propagation as the classifier.
- **SGC**: Wu et al., ICML 2019. https://arxiv.org/abs/1902.07153 . `ÂᴷX` then logistic regression, on a given graph.
- **S²GC / SSGC**: Zhu & Koniusz, ICLR 2021. https://iclr.cc/virtual/2021/poster/3377 *(arXiv id not verified)*.
- **APPNP**: Gasteiger et al., ICLR 2019. https://arxiv.org/abs/1810.05997 . Propagates a trained network's logits with PPR (the reverse order).
- **iLPC**: Lazarou, Stathaki, Avrithis, ICCV 2021. https://arxiv.org/abs/2012.07962 *(full text read)*. Symmetrized kNN graph on frozen-backbone embeddings, closed-form `(I − αW)⁻¹Y`, then pseudo-label cleaning. Labels only.
- **ZLaP**: Stojnić, Kalantidis, Tolias, CVPR 2024. https://arxiv.org/abs/2404.04072 *(full text read)*. Same operator on kNN graphs of CLIP features; zero-shot, uses class-name text nodes.
- **ECALP**: Li et al., ICLR 2025. https://arxiv.org/abs/2412.18303 *(text read)*. Label propagation on CLIP features with a *diagonal* re-weighting of feature dimensions derived from text prompts and few-shot labels. Motivated by VLM features encoding "objects, background, style". Conceptually the nearest thing to changing the graph metric for CLIP, but diagonal and not label-free.
- Also in the family: TPN (https://arxiv.org/abs/1805.10002), Correct & Smooth (https://arxiv.org/abs/2010.13993), SIGN (https://arxiv.org/abs/2004.11198), Iscen et al. 2017 / 2019 (https://arxiv.org/abs/1611.05113 , https://arxiv.org/abs/1904.04717), A2LP (https://arxiv.org/abs/2310.19996).

**scikit-learn's directed kNN graph.** Issue #11784, "Label Spreading: knn Kernel Variant differs from Reference Paper", open since 2018: https://github.com/scikit-learn/scikit-learn/issues/11784 . It notes the graph is not symmetric; no experiments, nothing about low label rates. No paper or post was found showing that this breaks few-label performance and that symmetrizing fixes it.

**Poisson learning.** Calder, Cook, Thorpe, Slepčev, ICML 2020. https://arxiv.org/abs/2006.11184 *(full text read)*. Their table at 1 label per class: MNIST — Laplace 16.1, Random Walk 66.4, Poisson 90.2; FashionMNIST — 18.4, 49.0, 60.8. Setup differs from ours (autoencoder features, 60–70k points, Gaussian-weighted graph). Any claim about label spreading versus Poisson learning has to be squared with this.

## 2. The cluster metric

**LDA on k-means labels / discriminative clustering** (the label-free step):

- **LDA-Km**: Ding & Li, "Adaptive Dimension Reduction Using Discriminant Analysis and K-means Clustering", ICML 2007. https://icml.cc/imls/conferences/2007/proceedings/abstracts/521.htm . Alternates k-means labels and an LDA subspace. For clustering; no graph.
- **Discriminative K-means**: Ye, Zhao, Wu, NIPS 2007. https://papers.nips.cc/paper/3176-discriminative-k-means-for-clustering .
- **Discriminative Cluster Analysis**: De la Torre & Kanade, ICML 2006. https://www.ri.cmu.edu/publications/discriminative-cluster-analysis .
- **DIFFRAC**: Bach & Harchaoui, NIPS 2007. https://papers.nips.cc/paper/2007/hash/22fb0cee7e1f3bde58293de743871417-Abstract.html *(search summary only)*. Clustering as regularized regression of the indicator matrix on the data: the closest match to "ridge onto cluster indicators".
- **Spectral regression / SRDA / semi-supervised discriminant analysis**: Cai, He, Han, ICCV 2007. https://mlanthology.org/iccv/2007/cai2007iccv-spectral , https://mlanthology.org/iccv/2007/cai2007iccv-semi *(summaries only)*. LDA solved as ridge regression.
- Regression onto indicators ↔ LDA: Hastie, Tibshirani, Buja 1994/95 *(abstract only)*. **Our matrix is not exactly LDA**: `WᵀW = Σ_λ⁻¹ XᵀY YᵀX Σ_λ⁻¹`, the between-cluster scatter sandwiched by the inverse regularized covariance *(derived by a reviewer, not from a paper)*. Whether it behaves like shrinkage LDA in practice is tested in `run_sens.py`.

**Metric learning to build the graph for semi-supervised learning:**

- **IDML**: Dhillon, Talukdar, Crammer, "Inference Driven Metric Learning for Graph Construction", UPenn tech report 2010, https://repository.upenn.edu/cis_reports/927 *(PDF not accessible)*; short version ACL 2010, https://aclanthology.org/P10-2069.pdf *(read)*. Loop: learn a Mahalanobis metric (ITML) on labeled + confidently propagated points → rebuild the kNN graph → propagate → repeat. **Structurally our pseudo-label variant.**
- Zhu, Ghahramani, Lafferty, ICML 2003. https://mlanthology.org/icml/2003/zhu2003icml-semi/ *(full text read)*: learns per-dimension edge-weight scales by minimizing label entropy.
- Ebert, Fritz, Schiele, DAGM 2011 *(summary only)*; Wauquier & Keller, https://arxiv.org/abs/1511.05789 *(summary only)*; CAN, Nie et al., KDD 2014 *(summary only)*.

**Gradient outer products as metrics** (the AGOP side):

- **Recursive feature machines**: Radhakrishnan, Beaglehole, Pandit, Belkin, Science 2024, https://arxiv.org/abs/2212.13881 *(preprint read)*. AGOP as a Mahalanobis matrix inside a kernel machine. Supervised. Lists "unsupervised and semi-supervised manifold learning" as a possible direction without doing it.
- **Trivedi, Wang, Kpotufe, Shakhnarovich**, "A consistent estimator of the expected gradient outerproduct", UAI 2014. https://home.ttic.edu/~shubhendu/Papers/GOP-UAI-2014.pdf *(read)*. Uses the gradient outer product explicitly as a metric for kNN and kernel regression / classification. Supervised.
- **Wu, Guinney, Maggioni, Mukherjee**, "Learning Gradients", JMLR 2010. https://www.jmlr.org/papers/v11/wu10a.html *(read)*. Gradients estimated from labeled points define graph edge weights; tested on the Chapelle semi-supervised benchmarks. Closest "gradient-derived graph for SSL"; uses true labels and a per-edge term, not a global metric.
- Kpotufe & Boularias, NIPS 2012 (diagonal gradient weights); Xia, Tong, Li, Zhu 2002 (MAVE / OPG); Hastie & Tibshirani 1996 (discriminant adaptive nearest neighbors).
- Recent, *not checked by us*: linear RFM (https://arxiv.org/abs/2401.04553), AGOP and neural collapse (https://arxiv.org/abs/2402.13728), xRFM (https://arxiv.org/abs/2508.10053), DANCE (https://arxiv.org/abs/2602.20652: supervised RFM on frozen CLIP embeddings + nearest neighbors).
- Searches for "unsupervised RFM", "semi-supervised recursive feature machine", "AGOP clustering", "AGOP pseudo-labels", "AGOP metric learning graph" found nothing.

**Label-free subspaces for transductive few-shot learning on frozen features:**

- **EASE**: Zhu & Koniusz, CVPR 2022 *(abstract only; page returned 403)*. Unsupervised discriminant subspace from support + query features. Not k-means-based, no graph rebuild.
- **TAFSSL**: Lichtenstein et al., ECCV 2020. https://arxiv.org/abs/2003.06670 . Task-adaptive PCA / ICA subspaces.
- Preprocessing family: SimpleShot (https://arxiv.org/abs/1911.04623), PT-MAP (https://arxiv.org/abs/2006.03806), hubness reduction (https://arxiv.org/abs/2303.09352).
- Other transductive few-shot methods, none matching: TIM, LaplacianShot, α-TIM, BD-CSPN, TransCLIP (https://arxiv.org/abs/2406.01837), Martin et al. (https://arxiv.org/abs/2405.18437), UNEM.

## 3. Why would it help CLIP and not DINOv2 or text?

What papers show:

- **CLIP image embeddings carry class-irrelevant variance.** Herzog & Wang, "Reevaluating the Intra-Modal Misalignment Hypothesis in CLIP", https://arxiv.org/abs/2603.16100 *(read by a reviewer; not checked by us)*. Image-only few-shot, no text prompts, 11 datasets: 1-shot prototype accuracy 43.4 (CLIP), 57.3 (SigLIP), 59.6 (DINOv2). Projecting CLIP image embeddings onto the principal subspace of class-name text embeddings lifts 1-shot from 43.4 to 50.7, and the gain fades with more shots. Our cluster metric is a label-free, text-free analogue, and shows the same pattern (about +5 at 1-shot, less at 5-shot).
- **kNN versus linear probe on ImageNet** (DINOv2 paper, https://arxiv.org/abs/2304.07193): CLIP ViT-L/14 79.8 vs 84.3; DINOv2 ViT-B/14 82.1 vs 84.5. CLIP's neighbor structure lags its linear separability more than DINOv2's does.
- **The "cone effect" is not the explanation**: Liang et al., "Mind the Gap", NeurIPS 2022, https://arxiv.org/abs/2203.02053 . Text embeddings sit in a cone too and show no gain.

Reviewer speculation, to be tested, not cited: DINOv2's image-image training suppresses style and background variation; CLIP's image-text training keeps whatever is describable. A cluster-supervised metric up-weights cluster-coherent directions, which are mostly semantic in CLIP.

Predictions this makes: the gain should appear for SigLIP and OpenCLIP; it should shrink with shots; per dataset it should track the gap between linear-probe and kNN accuracy; it should vanish or reverse when the labels are attributes instead of the dominant concept.

## 4. Benchmark conventions

- Classic few-shot: miniImageNet, tieredImageNet, CUB, CIFAR-FS; 5-way 1/5-shot, ~15 queries per class, mean ± 95% interval over many episodes.
- CLIP-era transductive work: the 11 CoOp-suite datasets, 1/2/4/8/16 shots, 75 queries, 1,000 tasks; class-imbalanced queries (Veilleux et al.) are now expected.
- Our pool of ~5,000 points with 10 seeds is larger than usual. Mean rank is our own choice, not a convention.

## Not verified by anyone

- Full text of IDML (tech report), EASE, DIFFRAC, Ebert et al., Xia 2002, Kpotufe 2016, Szlam 2008.
- Anything dated 2026 (DANCE, Herzog & Wang, xRFM's venue).
- That no 2025–26 workshop paper does unsupervised or semi-supervised AGOP.
- Hubness differences between CLIP and DINOv2 for image-image neighbors.
