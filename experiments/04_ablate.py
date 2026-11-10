"""
  04_ablate.py - stage 3: tuning + ablations of "targets -> ridge AGOP -> graph metric"

  usage (from repo root): python experiments/04_ablate.py <dataset> <seed> <out.csv>

  Reads data/<dataset>/... only (never downloads). Runs 1-shot and 5-shot in one process,
  because every label-free graph is shared between them.

  The method being ablated
    1. kNN graph on X (k)  ->  PPR-spread X (alpha)  ->  P
    2. targets Y: k-means clusters of P (K), and / or pseudo-labels from the few-shot classifier
    3. ridge-regress Y on P (reg)  ->  W ;  AGOP of that linear model = W'W
    4. metric M = (1 - lam) I + lam * W'W (trace-normalized) ; rebuild the kNN graph on X M^{1/2}
    5. re-spread X on the new graph -> P ; repeat from 2 for T rounds
    6. predict: LinearSVC on the labeled rows of P ('svc'), or label propagation on the graph ('lp')

  Rows: one per (method, config, round, npc, pred).
  homophily / oracle_* use ALL labels -- diagnostics only.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from time import time
from scipy.linalg import solve
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.exceptions import ConvergenceWarning
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from feature_spreading import MODEL
from feature_spreading.run import load_data
from feature_spreading.agop import metric_sqrt, damp_metric, agop_kernel, edge_homophily

warnings.filterwarnings('ignore', category=ConvergenceWarning)

dataset, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NPCS = [1, 5]
DEF  = dict(K=100, lam=0.5, reg=1.0, k=10, alpha=0.85, fit_on='P', graph_from='X')

# --
# Data (same sample for both shot counts; only the labeled split differs)

splits = {}
for npc in NPCS:
    X_, y_, (_, _, _, _, idx_train, idx_valid) = load_data(dataset, MODEL, 5_000, npc, seed)
    if not splits:
        X, y_raw = X_.astype(np.float64), y_
    assert (y_ == y_raw).all()
    splits[npc] = (idx_train, idx_valid)

_, y = np.unique(y_raw, return_inverse=True)
n, d = X.shape
C    = y.max() + 1

rows = []
def log(method, npc, pred, scores=None, yhat=None, **kw):
    idx_valid = splits[npc][1]
    yhat      = scores.argmax(1) if yhat is None else yhat
    rows.append(dict(dataset=dataset, seed=seed, npc=npc, method=method, pred=pred,
                     acc=(yhat[idx_valid] == y[idx_valid]).mean(), **kw))

# --
# Building blocks

def knn_graph(Xg, k=10):
    sim    = Xg @ Xg.T
    thresh = np.partition(sim, -k, axis=-1)[:,-k]
    adj    = (sim > thresh[:,None])
    return (adj | adj.T).astype(float)

def spread(adj, B, alpha=0.85):
    A    = adj + np.eye(n)
    dinv = 1 / np.sqrt(A.sum(1))
    A    = np.eye(n) - alpha * (A * dinv[:,None] * dinv[None,:])
    return (1 - alpha) * solve(A, B, assume_a='pos')

def fit_svc(F, npc):
    idx_train = splits[npc][0]
    return LinearSVC().fit(F[idx_train], y[idx_train]).decision_function(F)

def pm_onehot(lab, m):
    Y = -np.ones((len(lab), m))
    Y[np.arange(len(lab)), lab] = 1
    return Y

def ridge_W(F, Y, reg=1.0, idx=None):
    if idx is not None:
        F = F[idx]
    Fc = F - F.mean(0)
    return solve(Fc.T @ Fc + reg * np.eye(F.shape[1]), Fc.T @ (Y - Y.mean(0)), assume_a='pos').T

def kmeans(F, K):
    return KMeans(n_clusters=K, n_init=1, max_iter=100, random_state=seed).fit(F).labels_

def regraph(G, lam, k=10, alpha=0.85, src=None):
    src = X if src is None else src
    adj = knn_graph(normalize(src @ metric_sqrt(damp_metric(G, lam))), k)
    return adj, normalize(spread(adj, X, alpha))

def evaluate(method, adj, P, npcs=NPCS, lp_alpha=0.95, **cfg):
    """ svc on spread features + label propagation on the graph, for each shot count """
    hom = edge_homophily(adj, y)
    Y   = np.zeros((n, C * len(npcs)))
    for i, npc in enumerate(npcs):
        idx_train = splits[npc][0]
        Y[idx_train, i * C + y[idx_train]] = 1

    F      = spread(adj, Y, lp_alpha)
    scores = {}
    for i, npc in enumerate(npcs):
        scores[npc] = fit_svc(P, npc)
        log(method, npc, 'svc', scores[npc],               homophily=hom, **cfg)
        log(method, npc, 'lp',  F[:, i * C:(i + 1) * C],   homophily=hom, **cfg)

    return scores

def pseudo(scores, npc, q=1.0):
    # per predicted class, keep the top-q fraction by margin; labeled points always kept w/ true label
    idx_train = splits[npc][0]
    pred      = scores.argmax(1)
    part      = np.partition(scores, -2, axis=1)
    margin    = part[:,-1] - part[:,-2]
    keep      = np.zeros(n, dtype=bool)
    for c in range(C):
        ii = np.where(pred == c)[0]
        if len(ii):
            keep[ii[np.argsort(-margin[ii])[:int(np.ceil(q * len(ii)))]]] = True
    pred = pred.copy()
    pred[idx_train] = y[idx_train]
    keep[idx_train] = True
    idx = np.where(keep)[0]
    return idx, pred[idx]

# --
# 0. Baselines, ceilings

t0 = time()

base = {}
def get_base(k, alpha):
    if (k, alpha) not in base:
        adj = knn_graph(X, k)
        base[(k, alpha)] = (adj, normalize(spread(adj, X, alpha)))
    return base[(k, alpha)]

adj0, P0 = get_base(10, 0.85)
scores0  = evaluate('ppr', adj0, P0, k=10, alpha=0.85)

for npc in NPCS:
    log('svc', npc, 'svc', fit_svc(X, npc))

# label propagation alpha, on the base graph
for lp_alpha in [0.5, 0.85, 0.99]:
    Y = np.zeros((n, C * len(NPCS)))
    for i, npc in enumerate(NPCS):
        Y[splits[npc][0], i * C + y[splits[npc][0]]] = 1
    F = spread(adj0, Y, lp_alpha)
    for i, npc in enumerate(NPCS):
        log('lp_alpha', npc, 'lp', F[:, i * C:(i + 1) * C], lp_alpha=lp_alpha)

# sklearn's label propagation / spreading, as in the repo's original (commented-out) baseline
for npc in NPCS:
    idx_train = splits[npc][0]
    y_ss      = -np.ones(n, dtype=int)
    y_ss[idx_train] = y[idx_train]

    m = LabelPropagation(kernel='knn', n_neighbors=10, max_iter=2_000).fit(X, y_ss)
    log('sk_LabelPropagation', npc, 'lp', yhat=m.transduction_, n_iter=m.n_iter_)
    for a in [0.2, 0.85, 0.95]:
        m = LabelSpreading(kernel='knn', n_neighbors=10, alpha=a, max_iter=2_000).fit(X, y_ss)
        log('sk_LabelSpreading', npc, 'lp', yhat=m.transduction_, n_iter=m.n_iter_, lp_alpha=a)

# ceilings
adj = adj0 * (y[:,None] == y[None,:])
evaluate('oracle_graph', adj, normalize(spread(adj, X)))

W      = ridge_W(P0, pm_onehot(y, C))
adj, P = regraph(W.T @ W, 1.0)
evaluate('oracle_agop', adj, P)

print(f'{dataset} seed={seed} | baselines {time() - t0:.0f}s', flush=True)

# --
# 1. k-means AGOP: one-factor-at-a-time sweeps around DEF  (no labels in the graph)

def run_kmeans(T, sweep, **over):
    c      = {**DEF, **over}
    adj, P = get_base(c['k'], c['alpha'])
    for it in range(1, T + 1):
        F      = P if c['fit_on'] == 'P' else X
        W      = ridge_W(F, pm_onehot(kmeans(F, c['K']), c['K']), c['reg'])
        adj, P = regraph(W.T @ W, c['lam'], c['k'], c['alpha'], src=P if c['graph_from'] == 'P' else X)
        evaluate('kmeans_agop', adj, P, sweep=sweep, it=it, **c)

t0 = time()
run_kmeans(4, 'default')

for K in sorted(set([10, 20, 50, 200, 400, C]) - {DEF['K']}):
    run_kmeans(2, 'K', K=K)

for lam in [0.1, 0.25, 0.75, 1.0]:
    run_kmeans(2, 'lam', lam=lam)

for reg in [0.01, 0.1, 10.0]:
    run_kmeans(2, 'reg', reg=reg)

for K, lam in [(20, 1.0), (400, 1.0)]:
    run_kmeans(2, 'K_lam', K=K, lam=lam)

for k, alpha in [(5, 0.85), (20, 0.85), (10, 0.5), (10, 0.95)]:
    adj, P = get_base(k, alpha)
    evaluate('ppr', adj, P, k=k, alpha=alpha)
    run_kmeans(2, 'k_alpha', k=k, alpha=alpha)

# ablation: cluster + fit the metric on raw X (no smoothing anywhere before the metric)
run_kmeans(1, 'fit_on', fit_on='X')

# ablation: build the new graph from the smoothed features instead of the raw ones
run_kmeans(2, 'graph_from', graph_from='P')

print(f'{dataset} seed={seed} | kmeans sweeps {time() - t0:.0f}s', flush=True)

# --
# 2. Controls

# "is it just re-graphing on the smoothed features?"  (no metric at all)
adj, P = adj0, P0
for it in [1, 2]:
    adj = knn_graph(P)
    P   = normalize(spread(adj, X))
    evaluate('regraph_P', adj, P, it=it)

# random targets w/ the default number of classes: does the metric need real structure?
rng    = np.random.default_rng(seed)
W      = ridge_W(P0, pm_onehot(rng.integers(0, DEF['K'], n), DEF['K']))
adj, P = regraph(W.T @ W, DEF['lam'])
evaluate('random_agop', adj, P)

# same subspace as k-means (top-K principal directions of P0), no clustering / regression
Pc       = P0 - P0.mean(0)
_, S, Vt = np.linalg.svd(Pc, full_matrices=False)
r        = min(DEF['K'], d)
adj, P   = regraph(Vt[:r].T @ Vt[:r], DEF['lam'])
evaluate('pca_proj', adj, P)
adj, P   = regraph((Vt[:r].T / S[:r] ** 2) @ Vt[:r], DEF['lam'])
evaluate('pca_white', adj, P)

# --
# 3. Label-dependent variants (labels enter only through the few labeled points)

t0   = time()
Y_km = pm_onehot(kmeans(P0, DEF['K']), DEF['K'])
for npc in NPCS:
    # pseudo-labels from the few-shot classifier
    scores, adj, P = scores0[npc], adj0, P0
    for it in range(1, 4):
        idx_pl, y_pl = pseudo(scores, npc)
        W            = ridge_W(P, pm_onehot(y_pl, C), idx=idx_pl)
        adj, P       = regraph(W.T @ W, DEF['lam'])
        scores       = evaluate('pl_agop', adj, P, npcs=[npc], it=it)[npc]

    # pseudo-labels + k-means clusters as joint targets
    scores, adj, P = scores0[npc], adj0, P0
    for it in range(1, 4):
        idx_pl, y_pl = pseudo(scores, npc)
        Y            = np.column_stack([pm_onehot(y_pl, C), pm_onehot(kmeans(P, DEF['K']), DEF['K'])])
        W            = ridge_W(P, Y)
        adj, P       = regraph(W.T @ W, DEF['lam'])
        scores       = evaluate('pl+kmeans_agop', adj, P, npcs=[npc], it=it)[npc]

    # kernel AGOP (the full RFM step) on confident pseudo-labels -- only where affordable
    if C <= 50:
        scores, adj, P, M = scores0[npc], adj0, P0, np.eye(d)
        for it in range(1, 3):
            idx_pl, y_pl = pseudo(scores, npc, q=0.2)
            M            = damp_metric(agop_kernel(P, P[idx_pl], y_pl, M, reg=1e-1, bs=64), DEF['lam'])
            adj          = knn_graph(normalize(X @ metric_sqrt(M)))
            P            = normalize(spread(adj, X))
            scores       = evaluate('pl_kernel_agop', adj, P, npcs=[npc], it=it)[npc]

print(f'{dataset} seed={seed} | label-dependent {time() - t0:.0f}s', flush=True)

pd.DataFrame(rows).to_csv(out, index=False)
print(f'{dataset} seed={seed} | done -> {out}', flush=True)
