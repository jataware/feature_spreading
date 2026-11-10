"""
  explore2.py - stage 2: follow-ups on "pseudo-label / cluster -> ridge AGOP -> graph"

  usage: python explore2.py <dataset> <n_per_class> <seed> <out.csv>

  Reads data/<dataset>/... only (never downloads).
  All final classifiers are LinearSVC on the labeled points only (the repo's protocol).
  homophily / pl_acc / oracle_* use ALL labels -- diagnostics only.
"""

import sys
import numpy as np
import pandas as pd
from time import time
from scipy.linalg import solve
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from feature_spreading import MODEL
from feature_spreading.run import load_data
from feature_spreading.agop import metric_sqrt, damp_metric, edge_homophily

dataset, npc, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ALPHA = 0.85
K     = 10
T     = 4

X, y, (_, _, _, _, idx_train, idx_valid) = load_data(dataset, MODEL, 5_000, npc, seed)
X       = X.astype(np.float64)
_, y    = np.unique(y, return_inverse=True)
y_tr    = y[idx_train]
n, d    = X.shape
C       = y.max() + 1

rows = []
def log(method, scores=None, adj=None, **kw):
    r = dict(dataset=dataset, npc=npc, seed=seed, method=method, **kw)
    if scores is not None:
        r['acc'] = (scores.argmax(1)[idx_valid] == y[idx_valid]).mean()
    if adj is not None:
        r['homophily'] = edge_homophily(adj, y)
    rows.append(r)

def knn_graph(Xg, k=K):
    sim    = Xg @ Xg.T
    thresh = np.partition(sim, -k, axis=-1)[:,-k]
    adj    = (sim > thresh[:,None])
    return (adj | adj.T).astype(float)

def spread(adj, B, alpha=ALPHA):
    A    = adj + np.eye(n)
    dinv = 1 / np.sqrt(A.sum(1))
    A    = np.eye(n) - alpha * (A * dinv[:,None] * dinv[None,:])
    return (1 - alpha) * solve(A, B, assume_a='pos')

def fit_svc(F):
    return LinearSVC().fit(F[idx_train], y_tr).decision_function(F)

def ridge_W(F, idx, yl, n_out, reg=1.0):
    Fc = F[idx] - F[idx].mean(0)
    Y  = -np.ones((len(idx), n_out))
    Y[np.arange(len(idx)), yl] = 1
    return solve(Fc.T @ Fc + reg * np.eye(F.shape[1]), Fc.T @ (Y - Y.mean(0)), assume_a='pos').T

def pseudo(scores, q):
    pred   = scores.argmax(1)
    part   = np.partition(scores, -2, axis=1)
    margin = part[:,-1] - part[:,-2]
    keep   = np.zeros(n, dtype=bool)
    for c in range(C):
        ii = np.where(pred == c)[0]
        if len(ii):
            keep[ii[np.argsort(-margin[ii])[:int(np.ceil(q * len(ii)))]]] = True
    pred = pred.copy()
    pred[idx_train] = y_tr
    keep[idx_train] = True
    idx = np.where(keep)[0]
    return idx, pred[idx]

def regraph(G, lam):
    adj = knn_graph(normalize(X @ metric_sqrt(damp_metric(G, lam))))
    return adj, normalize(spread(adj, X))

def lp(adj):
    Y = np.zeros((n, C))
    Y[idx_train, y_tr] = 1
    return spread(adj, Y, 0.95)

# --
# Baselines + ceilings

s_svc   = fit_svc(X)
adj0    = knn_graph(X)
P0      = normalize(spread(adj0, X))
scores0 = fit_svc(P0)

log('svc', s_svc)
log('ppr', scores0, adj0)
log('lp',  lp(adj0))

adj = adj0 * (y[:,None] == y[None,:])
log('oracle_graph', fit_svc(normalize(spread(adj, X))), adj)

W = ridge_W(P0, np.arange(n), y, C)
adj, P = regraph(W.T @ W, 1.0)
log('oracle_agop', fit_svc(P), adj)

# --
# A. pseudo-label -> ridge AGOP -> graph   (labels enter only through y_tr)

def run_pl(q, lam, reg=1.0):
    scores, adj, P = scores0, adj0, P0
    for it in range(1, T + 1):
        idx_pl, y_pl = pseudo(scores, q)
        W            = ridge_W(P, idx_pl, y_pl, C, reg=reg)
        adj, P       = regraph(W.T @ W, lam)
        scores       = fit_svc(P)
        cfg = dict(q=q, lam=lam, reg=reg, it=it, pl_acc=(y_pl == y[idx_pl]).mean())
        log('pl_agop', scores, adj, **cfg)
        if it == T:
            log('pl_agop>lp', lp(adj), **cfg)

t0 = time()
for q in [0.5, 1.0]:
    for lam in [0.5, 1.0]:
        run_pl(q, lam)

for reg in [0.1, 10.0]:
    run_pl(0.5, 1.0, reg=reg)

print(f'{dataset} npc={npc} seed={seed} | pl_agop {time() - t0:.0f}s', flush=True)

# --
# B. k-means clusters -> ridge AGOP -> graph   (metric uses NO labels; recluster each round)

t0 = time()
for kmult in [1, 2]:
    for lam in [0.5, 1.0]:
        adj, P = adj0, P0
        for it in range(1, T + 1):
            km     = KMeans(n_clusters=kmult * C, n_init=1, random_state=seed).fit(P).labels_
            W      = ridge_W(P, np.arange(n), km, kmult * C)
            adj, P = regraph(W.T @ W, lam)
            log('kmeans_agop', fit_svc(P), adj, kmult=kmult, lam=lam, it=it)
            if it == T:
                log('kmeans_agop>lp', lp(adj), kmult=kmult, lam=lam, it=it)

print(f'{dataset} npc={npc} seed={seed} | kmeans_agop {time() - t0:.0f}s', flush=True)

# --
# C. Controls: is it "just whitening the smoothed features"?  (no labels, no clusters)

Pc       = P0 - P0.mean(0)
_, S, Vt = np.linalg.svd(Pc, full_matrices=False)
for r in [C, 2 * C]:
    r = min(r, d)
    # metric that whitens the top-r principal directions of P0 and drops the rest
    G      = (Vt[:r].T / S[:r] ** 2) @ Vt[:r]
    adj, P = regraph(G, 1.0)
    log('pca_white', fit_svc(P), adj, r=r)
    # same subspace, no whitening
    G      = Vt[:r].T @ Vt[:r]
    adj, P = regraph(G, 1.0)
    log('pca_proj', fit_svc(P), adj, r=r)

# random labels with the same number of classes: does the metric need *any* structure in Y?
rng    = np.random.default_rng(seed)
W      = ridge_W(P0, np.arange(n), rng.integers(0, C, n), C)
adj, P = regraph(W.T @ W, 1.0)
log('random_agop', fit_svc(P), adj)

pd.DataFrame(rows).to_csv(out, index=False)
print(f'{dataset} npc={npc} seed={seed} | done -> {out}', flush=True)
