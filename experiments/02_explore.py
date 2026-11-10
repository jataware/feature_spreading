"""
  explore.py - method-space exploration around {svc, ppr, agop}

  usage: python explore.py <dataset> <n_per_class> <seed> <out.csv> [full|core]

  Reads data/<dataset>/... only (never downloads).

  Conventions
    fit='lab' : final classifier = LinearSVC on the labeled points only (the repo's protocol)
    fit='pl'  : final classifier = ridge on labeled + pseudo-labeled points (self-training)
    homophily / pl_acc / oracle_* rows use ALL labels -- diagnostics only
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
from feature_spreading.agop import metric_sqrt, damp_metric, agop_kernel, edge_homophily

dataset, npc, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
GRID  = sys.argv[5] if len(sys.argv) > 5 else 'full'
ALPHA = 0.85
K     = 10

# --
# Data

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

# --
# Building blocks

def knn_graph(Xg, k=K, weighted=False):
    # same as feature_spreading.run.compute_knn_graph, + optional similarity weights
    sim    = Xg @ Xg.T
    thresh = np.partition(sim, -k, axis=-1)[:,-k]
    adj    = (sim > thresh[:,None])
    adj    = (adj | adj.T).astype(float)
    if weighted:
        adj = adj * np.clip(sim, 0, None)
    return adj

def spread(adj, B, alpha=ALPHA):
    # == compute_ppr(adj, alpha) @ B, via a solve instead of an explicit inverse
    A    = adj + np.eye(n)
    dinv = 1 / np.sqrt(A.sum(1))
    A    = np.eye(n) - alpha * (A * dinv[:,None] * dinv[None,:])
    return (1 - alpha) * solve(A, B, assume_a='pos')

def fit_svc(F):
    svc = LinearSVC().fit(F[idx_train], y_tr)
    return svc.decision_function(F)

def fit_ridge(F, idx, yl, reg=1.0):
    Fi    = F[idx]
    mu    = Fi.mean(0)
    Fc    = Fi - mu
    Y     = -np.ones((len(idx), C))
    Y[np.arange(len(idx)), yl] = 1
    Ym    = Y.mean(0)
    W     = solve(Fc.T @ Fc + reg * np.eye(F.shape[1]), Fc.T @ (Y - Ym), assume_a='pos').T # (C, d)
    return (F - mu) @ W.T + Ym, W

def pseudo(scores, q):
    # per predicted class, keep the top-q fraction by margin; labeled points always kept w/ true label
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

def onehot(idx, yl):
    Y = np.zeros((n, C))
    Y[idx, yl] = 1
    return Y

# --
# 0. Baselines

t0      = time()
s_svc   = fit_svc(X)
log('svc', s_svc)

adj0    = knn_graph(X)
P0      = normalize(spread(adj0, X))
scores0 = fit_svc(P0)
log('ppr', scores0, adj0)

# --
# 1. What gets propagated (same graph)

for alpha in [0.85, 0.95]:
    log('lp',          spread(adj0, onehot(idx_train, y_tr), alpha), alpha=alpha)   # labels
    log('svc>ppr',     spread(adj0, s_svc,   alpha), alpha=alpha)                   # predictions (APPNP-ish)
    log('ppr>svc>ppr', spread(adj0, scores0, alpha), alpha=alpha)                   # features, then predictions (C&S-ish)

# --
# 2. Unsupervised graph fixes

Xc       = X - X.mean(0)
_, S, Vt = np.linalg.svd(Xc, full_matrices=False)
feats    = {
    'raw'    : X,
    'center' : normalize(Xc),
    'white'  : normalize(Xc @ Vt[:256].T / S[:256]),
}

for g_feat, s_feat in [('raw', 'raw'), ('center', 'raw'), ('center', 'center'), ('white', 'raw')]:
    for k in [5, 10, 20]:
        for weighted in [False, True]:
            adj = knn_graph(feats[g_feat], k=k, weighted=weighted)
            hom = edge_homophily(adj, y)
            for alpha in [0.5, 0.85, 0.95]:
                P = normalize(spread(adj, feats[s_feat], alpha))
                log('ppr_unsup', fit_svc(P), g_feat=g_feat, s_feat=s_feat, k=k, weighted=weighted, alpha=alpha, homophily=hom)

print(f'{dataset} npc={npc} seed={seed} | baselines + unsup {time() - t0:.0f}s', flush=True)

# --
# 3. Oracles (ceilings)

adj = adj0 * (y[:,None] == y[None,:])
log('oracle_graph', fit_svc(normalize(spread(adj, X))), adj)

for reg in [0.1, 1.0, 10.0]:
    _, W = fit_ridge(P0, np.arange(n), y, reg=reg)
    for lam in [0.5, 1.0]:
        adj = knn_graph(normalize(X @ metric_sqrt(damp_metric(W.T @ W, lam))))
        log('oracle_agop', fit_svc(normalize(spread(adj, X))), adj, lam=lam, reg=reg)

# --
# 4. Self-training, graph fixed (the control for everything below)

for q in [0.2, 0.5, 1.0]:
    scores = scores0
    for it in range(1, 7):
        idx_pl, y_pl = pseudo(scores, q)
        scores, _    = fit_ridge(P0, idx_pl, y_pl)
        log('selftrain', scores, q=q, it=it, fit='pl', pl_acc=(y_pl == y[idx_pl]).mean())

# --
# 5. Pseudo-label -> metric -> graph / features

def metric_from(kind, P, idx_pl, y_pl, M):
    if kind == 'linear':   # AGOP of a linear head: W'W
        _, W = fit_ridge(P, idx_pl, y_pl)
        return W.T @ W
    elif kind == 'ncm':    # control: between-class scatter of pseudo-class means (no classifier)
        Pi = P[idx_pl]
        Mu = np.stack([Pi[y_pl == c].mean(0) for c in range(C)])
        Mu = Mu - Mu.mean(0)
        return Mu.T @ Mu
    elif kind == 'kernel': # RFM step: kernel ridge on labeled + pseudo, gradients over all points
        return agop_kernel(P, P[idx_pl], y_pl, M, reg=1e-1, bs=64)

def run_metric(kind, q, lam, where, T=3):
    scores, adj, P, M = scores0, adj0, P0, np.eye(d)
    for it in range(1, T + 1):
        idx_pl, y_pl = pseudo(scores, q)
        M = damp_metric(metric_from(kind, P, idx_pl, y_pl, M), lam)
        L = metric_sqrt(M)

        if where in ('graph', 'both'):
            adj = knn_graph(normalize(X @ L))
            P   = normalize(spread(adj, X))

        F = normalize(P @ L) if where in ('feat', 'both') else P

        cfg = dict(kind=kind, q=q, lam=lam, where=where, it=it, pl_acc=(y_pl == y[idx_pl]).mean())
        log('pl_metric', fit_svc(F), adj, fit='lab', **cfg)
        scores, _ = fit_ridge(F, idx_pl, y_pl)
        log('pl_metric', scores,     adj, fit='pl',  **cfg)

t0 = time()
for kind in ['linear', 'ncm']:
    for q in [0.2, 0.5, 1.0]:
        for lam in [0.5, 1.0]:
            for where in ['graph', 'feat', 'both']:
                run_metric(kind, q, lam, where)

print(f'{dataset} npc={npc} seed={seed} | pl_metric {time() - t0:.0f}s', flush=True)

if GRID == 'full':
    t0 = time()
    for lam in [0.5, 1.0]:
        run_metric('kernel', 0.2, lam, 'graph')
    print(f'{dataset} npc={npc} seed={seed} | kernel {time() - t0:.0f}s', flush=True)

# --
# 6. Pseudo-label -> prune graph directly (the pseudo version of oracle_graph)

for q in [0.2, 0.5, 1.0]:
    scores = scores0
    for it in range(1, 4):
        idx_pl, y_pl = pseudo(scores, q)
        lab          = -np.ones(n, dtype=int)
        lab[idx_pl]  = y_pl
        known        = lab >= 0
        cross        = known[:,None] & known[None,:] & (lab[:,None] != lab[None,:])
        adj          = adj0 * ~cross
        P            = normalize(spread(adj, X))

        cfg = dict(q=q, it=it, pl_acc=(y_pl == y[idx_pl]).mean())
        log('pl_prune', fit_svc(P), adj, fit='lab', **cfg)
        scores, _ = fit_ridge(P, idx_pl, y_pl)
        log('pl_prune', scores,     adj, fit='pl',  **cfg)

# --
# 7. Fully unsupervised metric: k-means clusters as "labels" -> linear AGOP -> graph

km   = KMeans(n_clusters=C, n_init=1, random_state=seed).fit(P0).labels_
_, W = fit_ridge(P0, np.arange(n), km)
for lam in [0.5, 1.0]:
    adj = knn_graph(normalize(X @ metric_sqrt(damp_metric(W.T @ W, lam))))
    log('kmeans_agop', fit_svc(normalize(spread(adj, X))), adj, lam=lam)

pd.DataFrame(rows).to_csv(out, index=False)
print(f'{dataset} npc={npc} seed={seed} | done -> {out}', flush=True)
