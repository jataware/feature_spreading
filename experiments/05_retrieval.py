"""
  05_retrieval.py - one-class "retrieval": given 1 (or 5) exemplars of ONE class, rank everything else

  usage (from repo root): python experiments/05_retrieval.py <dataset> <seed> <out.csv>

  Reads data/<dataset>/... only (never downloads).

  Setting
    - a query = npc exemplars of a single class. Nothing else is labeled: no negatives, no other classes,
      and the number of classes is unknown to the method.
    - every other point gets a score; we report average precision (AP) and precision@10 over the ranking,
      averaged over >= 100 queries per (dataset, seed, npc).

  Scorers
    cos      cosine similarity to the (mean) exemplar
    elda     one-vs-rest ridge w/ every other point as a negative (closed form; "exemplar LDA")
    svc1vr   one-vs-rest LinearSVC: exemplars vs 500 random unlabeled points
    aqe      average query expansion: add the top-m neighbors to the query, re-rank
    ppr      personalized PageRank from the exemplars on the kNN graph (== label propagation w/ one class)

  ...each on raw X, or on features spread over a graph (base kNN, re-graphed, k-means-AGOP metric, oracle).
  oracle_* rows use ALL labels -- ceilings only.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from time import time
from scipy.linalg import solve, cho_factor, cho_solve
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.exceptions import ConvergenceWarning

from feature_spreading import MODEL
from feature_spreading.run import load_data
from feature_spreading.agop import metric_sqrt, damp_metric, edge_homophily

warnings.filterwarnings('ignore', category=ConvergenceWarning)

dataset, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# skip work another worker already has (lets several worker pools share one job list)
if os.path.exists(out) or os.path.exists(out + '.lock'):
    sys.exit(0)
open(out + '.lock', 'w').close()
NPCS = [1, 5]

X, y, _ = load_data(dataset, MODEL, 5_000, 1, seed)
X       = X.astype(np.float64)
_, y    = np.unique(y, return_inverse=True)
n, d    = X.shape
C       = y.max() + 1
rng     = np.random.default_rng(seed)

# --
# Queries: for each class, npc random exemplars; repeated so there are >= 100 queries

queries = {}
for npc in NPCS:
    Yq, qc = [], []
    for rep in range(int(np.ceil(100 / C))):
        for c in range(C):
            ii = np.where(y == c)[0]
            if len(ii) < npc + 5:
                continue
            yq = np.zeros(n)
            yq[rng.choice(ii, npc, replace=False)] = 1
            Yq.append(yq)
            qc.append(c)

    queries[npc] = (np.column_stack(Yq), np.array(qc))

Yall  = np.column_stack([queries[npc][0] for npc in NPCS])
slc   = {}
start = 0
for npc in NPCS:
    slc[npc] = slice(start, start + queries[npc][0].shape[1])
    start    = slc[npc].stop

def metrics(S, Yq, qc):
    S     = np.where(Yq > 0, -np.inf, S)                      # exemplars are not retrievable
    R     = (y[:,None] == qc[None,:]) & (Yq == 0)
    order = np.argsort(-S, axis=0, kind='stable')
    Rs    = np.take_along_axis(R, order, 0)
    prec  = np.cumsum(Rs, 0) / np.arange(1, n + 1)[:,None]
    return (prec * Rs).sum(0) / R.sum(0), Rs[:10].mean(0)

rows = []
def log(graph, scorer, S, **kw):
    """ S: (n, n_queries_total) scores, columns ordered as Yall """
    for npc in NPCS:
        Yq, qc  = queries[npc]
        ap, p10 = metrics(S[:, slc[npc]], Yq, qc)
        rows.append(dict(dataset=dataset, seed=seed, npc=npc, graph=graph, scorer=scorer,
                         map=ap.mean(), p10=p10.mean(), nq=len(qc), **kw))

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

def s_cos(F, Y=Yall):
    return F @ (F.T @ Y)

def s_elda(F, reg=1.0, Y=Yall):
    Fc = F - F.mean(0)
    return Fc @ solve(Fc.T @ Fc + reg * np.eye(F.shape[1]), Fc.T @ (Y - Y.mean(0)), assume_a='pos')

neg_pool = rng.choice(n, 500, replace=False)
def s_svc1vr(F, Y=Yall):
    S = np.zeros(Y.shape)
    for j in range(Y.shape[1]):
        pos    = np.where(Y[:,j] > 0)[0]
        neg    = np.setdiff1d(neg_pool, pos)
        idx    = np.concatenate([pos, neg])
        lab    = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        S[:,j] = LinearSVC(class_weight='balanced').fit(F[idx], lab).decision_function(F)
    return S

def s_aqe(F, m=10, Y=Yall):
    S   = np.where(Y > 0, -np.inf, s_cos(F, Y))
    top = np.argpartition(-S, m, axis=0)[:m]                  # (m, n_queries)
    Q   = F.T @ Y + F[top].sum(0).T
    return F @ Q

def pm_onehot(lab, m):
    Y = -np.ones((len(lab), m))
    Y[np.arange(len(lab)), lab] = 1
    return Y

def ridge_W(F, Y, reg=1.0):
    Fc = F - F.mean(0)
    return solve(Fc.T @ Fc + reg * np.eye(F.shape[1]), Fc.T @ (Y - Y.mean(0)), assume_a='pos').T

def kmeans(F, K):
    return KMeans(n_clusters=K, n_init=1, max_iter=100, random_state=seed).fit(F).labels_

def eval_graph(name, adj, svc=False, **kw):
    P  = normalize(spread(adj, X))
    kw = dict(homophily=edge_homophily(adj, y), **kw)
    log(name, 'cos',  s_cos(P),  **kw)
    log(name, 'elda', s_elda(P), **kw)
    for alpha in [0.85, 0.95, 0.99]:
        log(name, 'ppr', spread(adj, Yall, alpha), ppr_alpha=alpha, **kw)
    if svc:
        log(name, 'svc1vr', s_svc1vr(P), **kw)
    return P

# --
# 0. No graph

t0 = time()
log('none', 'cos',    s_cos(X))
log('none', 'svc1vr', s_svc1vr(X))
for reg in [0.1, 1.0, 10.0]:
    log('none', 'elda', s_elda(X, reg), reg=reg)
for m in [5, 10, 20]:
    log('none', 'aqe', s_aqe(X, m), m=m)

# --
# 1. Base graph, k sweep

adj0 = knn_graph(X)
P0   = eval_graph('knn', adj0, svc=True, k=10)
for k in [5, 20]:
    eval_graph('knn', knn_graph(X, k), k=k)

for m in [5, 10, 20]:
    log('knn', 'aqe', s_aqe(P0, m), m=m, k=10)

# --
# 2. Label-free learned graphs (query-independent: computed once, shared by every query)

# re-graph on the smoothed features (no metric)
eval_graph('regraph_P', knn_graph(P0))

def kmeans_metric(F, K, lam, reg=1.0):
    W = ridge_W(F, pm_onehot(kmeans(F, K), K), reg)
    return metric_sqrt(damp_metric(W.T @ W, lam))

for K in [20, 50, 100, 200, 400]:
    L   = kmeans_metric(P0, K, 0.5)
    adj = knn_graph(normalize(X @ L))
    P   = eval_graph('kmeans_agop', adj, svc=(K == 100), K=K, lam=0.5, it=1)
    if K == 100:
        # ablation: the metric alone, no graph / diffusion
        log('none', 'cos_metric', s_cos(normalize(X @ L)), K=K, lam=0.5)
        # second round
        eval_graph('kmeans_agop', knn_graph(normalize(X @ kmeans_metric(P, K, 0.5))), K=K, lam=0.5, it=2)

eval_graph('kmeans_agop', knn_graph(normalize(X @ kmeans_metric(P0, 100, 1.0))), K=100, lam=1.0, it=1)

# cluster + fit the metric on raw X (no smoothing before the metric)
eval_graph('kmeans_agop_rawfit', knn_graph(normalize(X @ kmeans_metric(X, 100, 0.5))), K=100, lam=0.5, it=1)

# --
# 3. Ceilings

eval_graph('oracle_graph', adj0 * (y[:,None] == y[None,:]))
W = ridge_W(P0, pm_onehot(y, C))
eval_graph('oracle_agop', knn_graph(normalize(X @ metric_sqrt(damp_metric(W.T @ W, 1.0)))))

print(f'{dataset} seed={seed} | shared graphs {time() - t0:.0f}s', flush=True)

# --
# 4. Query-specific metric (the retrieval analogue of pseudo-label AGOP), on 20 queries per shot count
#
#    pseudo-positives = exemplars + top-m by cos(P0);  ridge 1-vs-rest on P0 -> w ;  AGOP = w w' (rank 1)
#    M = (1 - lam) I + lam * d * u u'   (u = w / |w|)   ->   per-query graph  ->  re-spread  ->  rank

t0   = time()
sub  = {npc : rng.choice(queries[npc][0].shape[1], 20, replace=False) for npc in NPCS}
Ysub = np.column_stack([queries[npc][0][:, sub[npc]] for npc in NPCS])
slc_sub, start = {}, 0
for npc in NPCS:
    slc_sub[npc] = slice(start, start + 20)
    start += 20

def log_sub(graph, scorer, S, **kw):
    for npc in NPCS:
        Yq, qc  = queries[npc]
        ap, p10 = metrics(S[:, slc_sub[npc]], Yq[:, sub[npc]], qc[sub[npc]])
        rows.append(dict(dataset=dataset, seed=seed, npc=npc, graph=graph, scorer=scorer, subset='q20',
                         map=ap.mean(), p10=p10.mean(), nq=20, **kw))

# baselines on the same 20 queries
S0 = s_cos(P0, Ysub)
log_sub('knn', 'cos', S0)
log_sub('knn', 'ppr', spread(adj0, Ysub, 0.85), ppr_alpha=0.85)

Pc   = P0 - P0.mean(0)
chol = cho_factor(Pc.T @ Pc + np.eye(d))
for m in [20]:
    S_m  = np.where(Ysub > 0, -np.inf, S0)
    top  = np.argpartition(-S_m, m, axis=0)[:m]
    T    = Ysub.copy()
    for j in range(T.shape[1]):
        T[top[:,j], j] = 1
    Wq = cho_solve(chol, Pc.T @ (T - T.mean(0)))              # (d, n_queries)
    Wq = Wq / np.linalg.norm(Wq, axis=0)

    for lam in [0.02, 0.1]:
        a, b  = 1 - lam, lam * d
        S_cos = np.zeros(Ysub.shape)
        S_ppr = np.zeros(Ysub.shape)
        for j in range(Ysub.shape[1]):
            u    = Wq[:,j]
            XL   = np.sqrt(a) * X + (np.sqrt(a + b) - np.sqrt(a)) * np.outer(X @ u, u)
            adj  = knn_graph(normalize(XL))
            Z    = spread(adj, np.column_stack([X, Ysub[:,j]]))
            Pq   = normalize(Z[:,:d])
            S_cos[:,j] = Pq @ (Pq.T @ Ysub[:,j])
            S_ppr[:,j] = Z[:,d]

        log_sub('query_agop', 'cos', S_cos, lam=lam, m=m)
        log_sub('query_agop', 'ppr', S_ppr, lam=lam, m=m, ppr_alpha=0.85)

print(f'{dataset} seed={seed} | query-specific {time() - t0:.0f}s', flush=True)

pd.DataFrame(rows).to_csv(out, index=False)
os.remove(out + '.lock')
print(f'{dataset} seed={seed} | done -> {out}', flush=True)
