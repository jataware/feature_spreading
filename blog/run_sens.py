"""
  run_sens.py - sensitivity of the cluster-metric graph to each of its settings, one factor at a time
  
  usage (from repo root): python blog/run_sens.py <dataset> <seed>
  
  Around the frozen defaults (K=100, lam=0.5, penalty=2e-4 per sample, k=10, alpha 0.95 for labels / 0.85 for features):
  rebuild the graph with one thing changed, then score both predictors at 1-shot and 5-shot.
  Rows: factor, value, graph ('cmg' = cluster-metric, 'plain' = no metric), pred ('ls' label spreading, 'fs' features + SVM).
"""

import os
import sys
import numpy as np
import pandas as pd
from time import time
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import data

dataset, seed = sys.argv[1], int(sys.argv[2])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'sens', f'{dataset}_{seed}.csv')
if os.path.exists(out):
    sys.exit(0)

SHOTS = [1, 5]
DEF   = dict(K=100, lam=0.5, penalty=2e-4, k=10)

X, y = data.load(dataset)
X, y = lib.sample(X, y, n=5_000, min_count=2 * max(SHOTS), seed=seed)
ctx  = lib.Ctx(X, y, seed=seed)
X    = ctx.X
n, d = X.shape
lab  = {s : lib.draw_labeled(y, s, seed=seed) for s in SHOTS}
test = {s : np.setdiff1d(np.arange(n), lab[s]) for s in SHOTS}

rows = []
same = (y[:,None] == y[None,:])
def homophily(g):
    return float((g.adj * same).sum() / g.adj.sum())       # fraction of edges joining two points of the same class (diagnostic)

def score(factor, value, graph, g, alpha_ls=0.95, alpha_fs=0.85, shots=SHOTS, preds=('ls', 'fs')):
    hom = homophily(g)
    for s in shots:
        idx  = lab[s]
        base = dict(dataset=dataset, seed=seed, shots=s, factor=factor, value=value, graph=graph, homophily=hom)
        if 'ls' in preds:
            yhat = g.ppr(ctx.onehot(idx, y[idx]), alpha_ls).argmax(1)
            rows.append(dict(**base, pred='ls', acc=(yhat[test[s]] == y[test[s]]).mean()))
        if 'fs' in preds:
            P    = normalize(g.ppr(X, alpha_fs))
            yhat = LinearSVC().fit(P[idx], y[idx]).predict(P)
            rows.append(dict(**base, pred='fs', acc=(yhat[test[s]] == y[test[s]]).mean()))

def cmg(k=None, **kw):
    cfg = {**DEF, **kw}
    L   = lib.cluster_metric(X, cfg['K'], cfg['lam'], cfg['penalty'], cfg.get('seed', seed), targets=cfg.get('targets'), Y=cfg.get('Y'))
    return lib.Graph(lib.knn_graph(normalize(X @ L), k or cfg['k']))

def from_map(Z, k=10):
    return lib.Graph(lib.knn_graph(normalize(Z), k))

t0      = time()
g_plain = ctx.graph('plain')
g_def   = cmg()
score('default', 0, 'plain', g_plain)
score('default', 0, 'cmg',   g_def)

# what the vectors look like (label-free, except purity): helps compare encoders
Xc  = X - X.mean(0)
ev  = np.clip(np.linalg.eigvalsh(Xc.T @ Xc / n), 0, None)[::-1]
c_d = KMeans(n_clusters=DEF['K'], n_init=1, max_iter=100, random_state=seed).fit_predict(X)
pur = sum(np.bincount(y[c_d == j]).max() for j in range(DEF['K']) if (c_d == j).any()) / n
DIAG = dict(d=d, n=n, n_classes=ctx.C, mean_norm=float(np.linalg.norm(X.mean(0))),        # 1.0 = all vectors identical; 0 = centered
            mean_eig=float(ev.mean()), top_eig=float(ev[0]), eff_rank=float(ev.sum() ** 2 / (ev ** 2).sum()),
            cluster_purity=float(pur), cluster_size_min=int(np.bincount(c_d).min()), cluster_size_max=int(np.bincount(c_d).max()))

# -- single hyperparameters
for K in [10, 20, 50, 200, 400, 800]:
    score('K', K, 'cmg', cmg(K=K))
score('K', 100, 'cmg', g_def)

for lam in [0.1, 0.25, 0.75, 1.0]:
    score('lam', lam, 'cmg', cmg(lam=lam))
score('lam', 0.0, 'cmg', g_plain); score('lam', 0.5, 'cmg', g_def)

for pen in [2e-6, 2e-5, 6e-5, 6e-4, 2e-3, 6e-3, 2e-2, 6e-2, 2e-1]:
    score('penalty', pen, 'cmg', cmg(penalty=pen))
score('penalty', 2e-4, 'cmg', g_def)

for k in [5, 15, 20, 30, 50]:
    score('k', k, 'plain', lib.Graph(lib.knn_graph(X, k)))
    score('k', k, 'cmg',   cmg(k=k))
score('k', 10, 'plain', g_plain); score('k', 10, 'cmg', g_def)

for a in [0.5, 0.8, 0.9, 0.95, 0.99]:
    score('alpha_ls', a, 'plain', g_plain, alpha_ls=a, preds=('ls',))
    score('alpha_ls', a, 'cmg',   g_def,   alpha_ls=a, preds=('ls',))
for a in [0.5, 0.7, 0.85, 0.9, 0.95]:
    score('alpha_fs', a, 'plain', g_plain, alpha_fs=a, preds=('fs',))
    score('alpha_fs', a, 'cmg',   g_def,   alpha_fs=a, preds=('fs',))

print(f'{dataset} seed={seed} | hyperparameters {time() - t0:.0f}s', flush=True)

# -- centering: build graphs on mean-centered vectors
Xn = normalize(Xc)
score('center', 'graph on centered vectors', 'plain', lib.Graph(lib.knn_graph(Xn, DEF['k'])))
Lc = lib.cluster_metric(Xn, DEF['K'], DEF['lam'], DEF['penalty'], seed)
score('center', 'graph on centered vectors', 'cmg',   lib.Graph(lib.knn_graph(normalize(Xn @ Lc), DEF['k'])))

# -- where the metric is fit: raw vectors (default) or vectors already spread over the plain graph
P0s = normalize(g_plain.ppr(X, 0.85))
Ls  = lib.cluster_metric(P0s, DEF['K'], DEF['lam'], DEF['penalty'], seed)
score('fit_on', 'raw vectors (default)', 'cmg', g_def)
score('fit_on', 'spread vectors',        'cmg', lib.Graph(lib.knn_graph(normalize(X @ Ls), DEF['k'])))

# -- no metric at all: just rebuild the graph on the spread vectors
score('fit_on', 'no metric, graph on spread vectors', 'cmg', lib.Graph(lib.knn_graph(P0s, DEF['k'])))

# -- rounds: recluster on features spread over the current graph
g = g_def
score('rounds', 1, 'cmg', g)
for r in [2, 3]:
    c = KMeans(n_clusters=DEF['K'], n_init=1, max_iter=100, random_state=seed).fit_predict(normalize(g.ppr(X, 0.85)))
    g = cmg(targets=c)
    score('rounds', r, 'cmg', g)

# -- k-means: how much does the clustering itself vary?
for ks in range(1, 6):
    score('kmeans_seed', ks, 'cmg', cmg(seed=1000 + ks))
c = MiniBatchKMeans(n_clusters=DEF['K'], n_init=1, random_state=seed).fit_predict(X)
score('clusterer', 'minibatch_kmeans', 'cmg', cmg(targets=c))
score('clusterer', 'kmeans', 'cmg', g_def)

# -- what kind of metric? (same clusters where clusters are used)
c0 = KMeans(n_clusters=DEF['K'], n_init=1, max_iter=100, random_state=seed).fit_predict(X)
score('metric', 'ridge_agop (default)', 'cmg', g_def)
score('metric', 'none (plain graph)',   'cmg', g_plain)
score('metric', 'lda_shrinkage',        'cmg', from_map(LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto').fit(X, c0).transform(X)))
mu = np.stack([X[c0 == j].mean(0) for j in range(DEF['K'])]); mu = mu - mu.mean(0)
G  = mu.T @ mu
score('metric', 'cluster_means',        'cmg', from_map(X @ lib.sqrtm_psd(0.5 * np.eye(d) + 0.5 * G * d / np.trace(G))))
pca = PCA(n_components=min(DEF['K'], d - 1), random_state=seed).fit(X)
score('metric', 'pca_project',          'cmg', from_map(pca.transform(X)))
score('metric', 'pca_whiten',           'cmg', from_map(pca.transform(X) / np.sqrt(pca.explained_variance_)))
score('metric', 'random_targets',       'cmg', cmg(targets=np.random.default_rng(seed).integers(0, DEF['K'], n)))

# -- targets: clusters (default) vs pseudo-labels from the plain-graph classifier vs both
P0 = normalize(g_plain.ppr(X, 0.85))
for s in SHOTS:
    idx    = lab[s]
    pseudo = LinearSVC().fit(P0[idx], y[idx]).predict(P0)
    pseudo[idx] = y[idx]
    score('targets', 'clusters', 'cmg', g_def, shots=[s])
    score('targets', 'pseudo',   'cmg', cmg(targets=pseudo), shots=[s])
    score('targets', 'both',     'cmg', cmg(Y=np.column_stack([np.eye(ctx.C)[pseudo], np.eye(DEF['K'])[c0]])), shots=[s])

os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(rows).assign(**DIAG).to_csv(out, index=False)
print(f'{dataset} seed={seed} | done -> {out} ({time() - t0:.0f}s)', flush=True)
