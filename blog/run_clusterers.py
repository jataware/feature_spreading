"""
  run_clusterers.py - does the clustering algorithm matter for the cluster metric?
  
  usage (from repo root): python blog/run_clusterers.py <dataset> <seed>
  
  The cluster metric only uses clusters as regression targets: cluster -> ridge onto the indicators -> W'W -> graph.
  This swaps k-means for other scikit-learn clusterers and keeps everything else fixed (lam=0.5, penalty, k=10).
  Points a density-based method calls "noise" get an all-zero target row.
"""

import os
import sys
import numpy as np
import pandas as pd
from time import time
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, HDBSCAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import data

dataset, seed = sys.argv[1], int(sys.argv[2])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'clusterers', f'{dataset}_{seed}.csv')
if os.path.exists(out):
    sys.exit(0)

K     = 100
X, y  = data.load(dataset)
X, y  = lib.sample(X, y, seed=seed)
ctx   = lib.Ctx(X, y, seed=seed)
X, n  = ctx.X, ctx.n
plain = ctx.graph('plain')
d10   = np.sort(1 - X @ X.T, axis=1)[:, 10]            # cosine distance to the 10th neighbor

CLUSTERERS = {
    'kmeans (default)'        : lambda: KMeans(n_clusters=K, n_init=1, max_iter=100, random_state=seed).fit_predict(X),
    'bisecting kmeans'        : lambda: BisectingKMeans(n_clusters=K, random_state=seed).fit_predict(X),
    'agglomerative, ward'     : lambda: AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(X),
    'agglomerative, average'  : lambda: AgglomerativeClustering(n_clusters=K, linkage='average', metric='cosine').fit_predict(X),
    'spectral, on kNN graph'  : lambda: SpectralClustering(n_clusters=K, affinity='precomputed', assign_labels='kmeans', random_state=seed).fit_predict(sparse.csr_matrix(plain.adj)),
    'hdbscan'                 : lambda: HDBSCAN(min_cluster_size=10).fit_predict(X),
    'dbscan'                  : lambda: DBSCAN(eps=float(np.median(d10)), min_samples=10, metric='cosine').fit_predict(X),
}

rows = []
def score(name, g, secs, **info):
    for shots in [1, 5]:
        idx  = lib.draw_labeled(y, shots, seed=seed)
        test = np.setdiff1d(np.arange(n), idx)
        ls   = g.ppr(ctx.onehot(idx, y[idx]), 0.95).argmax(1)
        P    = normalize(g.ppr(X, 0.85))
        fs   = LinearSVC().fit(P[idx], y[idx]).predict(P)
        rows.append(dict(dataset=dataset, seed=seed, shots=shots, clusterer=name, secs=secs, **info,
                         ls=(ls[test] == y[test]).mean(), fs=(fs[test] == y[test]).mean()))

score('none (plain graph)', plain, 0.0, n_clusters=0, noise=0.0)
for name, fn in CLUSTERERS.items():
    t = time()
    try:
        c = fn()
    except Exception as e:
        print(f'!! {dataset} seed={seed} {name}: {type(e).__name__}: {e}', flush=True)
        continue
    secs  = time() - t
    k     = int(c.max()) + 1
    noise = float((c < 0).mean())
    if k < 2:
        rows += [dict(dataset=dataset, seed=seed, shots=s, clusterer=name, secs=secs, n_clusters=k, noise=noise, ls=np.nan, fs=np.nan) for s in [1, 5]]
        continue
    Y = np.zeros((n, k))
    Y[np.where(c >= 0)[0], c[c >= 0]] = 1
    L = lib.cluster_metric(X, lam=ctx.cfg['lam'], penalty=ctx.cfg['penalty'], Y=Y)
    score(name, lib.Graph(lib.knn_graph(normalize(X @ L), ctx.cfg['k'])), secs, n_clusters=k, noise=noise)

os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f'[clusterers] {dataset} seed={seed} -> {out}', flush=True)
