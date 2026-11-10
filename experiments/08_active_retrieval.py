"""
  08_active_retrieval.py - "active retrieval": retrieve a batch, get (noisy) labels for it, update, repeat

  usage (from repo root): python experiments/08_active_retrieval.py <dataset> <seed> <out.csv>

  Reads data/<dataset>/... only (never downloads).

  Loop (one target class per query)
    start : 1 clean exemplar of the target class; nothing else is labeled
    round : score every unseen point -> take the top B = 10 -> a labeler says relevant / not relevant for each,
            and is WRONG with probability `noise` (stand-in for an LLM judge) -> update the model -> repeat, T = 10 rounds

  What gets updated
    the query model only (cheap)       : rocchio / svm on a fixed embedding, or PPR seeds on a fixed graph
    the embedding of the DB (one-off)  : spread features / the label-free cluster-metric graph from 04 + 05, built once
                                          before any feedback and shared by every query. Per-query re-embedding from the
                                          feedback itself is NOT in this script.

  Update rules
    cos      no update: keep walking down the initial cosine ranking (baseline: what feedback buys)
    rocchio  query = mean(labeled relevant) - 0.5 * mean(labeled not-relevant); rank by dot product
    svm      LinearSVC: labeled relevant vs labeled not-relevant, plus 200 random unlabeled points as weak negatives
    ppr      personalized PageRank from the labeled points: +1 on relevant, -beta on not-relevant

  Logged per round, averaged over queries
    batch_prec  fraction of this round's 10 that are truly relevant (what the user sees)
    eff         relevant found so far / max possible so far  (= min(labels spent, class size))
    recall      relevant found so far / class size
    ap_unseen   average precision of the current ranking over the still-unseen points (quality of the model itself)
  Metrics use the TRUE labels; the methods only ever see the noisy ones.
"""

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
from feature_spreading.agop import metric_sqrt, damp_metric

warnings.filterwarnings('ignore', category=ConvergenceWarning)

dataset, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
Q, B, T = 60, 10, 10
NOISE   = [0.0, 0.1, 0.3]

X, y, _ = load_data(dataset, MODEL, 5_000, 1, seed)
X       = X.astype(np.float64)
_, y    = np.unique(y, return_inverse=True)
n, d    = X.shape
rng     = np.random.default_rng(seed)

# --
# Embeddings / graphs, all built once, before any feedback, without labels

def knn_graph(Xg, k=10):
    sim    = Xg @ Xg.T
    thresh = np.partition(sim, -k, axis=-1)[:,-k]
    adj    = (sim > thresh[:,None])
    return (adj | adj.T).astype(float)

def factor(adj, alpha):
    A    = adj + np.eye(n)
    dinv = 1 / np.sqrt(A.sum(1))
    return cho_factor(np.eye(n) - alpha * (A * dinv[:,None] * dinv[None,:]))

def cluster_metric(F, K=100, lam=0.5, reg=1.0):
    lab = KMeans(n_clusters=K, n_init=1, max_iter=100, random_state=seed).fit(F).labels_
    Y   = -np.ones((n, K))
    Y[np.arange(n), lab] = 1
    Fc  = F - F.mean(0)
    W   = solve(Fc.T @ Fc + reg * np.eye(d), Fc.T @ (Y - Y.mean(0)), assume_a='pos').T
    return metric_sqrt(damp_metric(W.T @ W, lam))

t0    = time()
adj0  = knn_graph(X)
P0    = normalize(cho_solve(factor(adj0, 0.85), X))            # features spread over the plain kNN graph
adjK  = knn_graph(normalize(X @ cluster_metric(P0)))
PK    = normalize(cho_solve(factor(adjK, 0.85), X))            # features spread over the cluster-metric graph
FEATS = {'raw' : X, 'spread' : P0, 'spread_km' : PK}
PPR   = {'knn' : factor(adj0, 0.95), 'km' : factor(adjK, 0.95)}
print(f'{dataset} seed={seed} | graphs {time() - t0:.0f}s', flush=True)

# --
# Queries: Q random exemplars from classes with at least 12 members

counts = np.bincount(y)
q_idx  = rng.choice(np.where(counts[y] >= 12)[0], Q, replace=False)
qc     = y[q_idx]
cols   = np.arange(Q)
rel    = (y[:,None] == qc[None,:])
rel[q_idx, cols] = False                                       # the exemplar itself is not retrievable
n_pos  = rel.sum(0)
bg     = rng.choice(n, 200, replace=False)                     # weak negatives for the svm

def ap_unseen(S, seen):
    R     = rel & ~seen
    order = np.argsort(-S, axis=0, kind='stable')              # seen points have S = -inf, so they sort last
    Rs    = np.take_along_axis(R, order, 0)
    prec  = np.cumsum(Rs, 0) / np.arange(1, n + 1)[:,None]
    tot   = R.sum(0)
    return np.where(tot > 0, (prec * Rs).sum(0) / np.maximum(tot, 1), np.nan)

# --
# Update rules: (pos, neg) are (n, Q) boolean matrices of the labels observed so far

def s_cos(F):
    S0 = F @ F[q_idx].T
    return lambda pos, neg: S0

def s_rocchio(F):
    def score(pos, neg):
        qp = F.T @ pos / pos.sum(0)
        qn = F.T @ neg / np.maximum(neg.sum(0), 1)
        return F @ (qp - 0.5 * qn)
    return score

def s_svm(F):
    def score(pos, neg):
        S = np.zeros((n, Q))
        for j in range(Q):
            ip, ineg = np.where(pos[:,j])[0], np.where(neg[:,j])[0]
            ib       = np.setdiff1d(bg, np.concatenate([ip, ineg]))
            idx      = np.concatenate([ip, ineg, ib])
            lab      = np.concatenate([np.ones(len(ip)), np.zeros(len(ineg) + len(ib))])
            w        = np.concatenate([np.ones(len(ip) + len(ineg)), 0.2 * np.ones(len(ib))])
            S[:,j]   = LinearSVC(class_weight='balanced').fit(F[idx], lab, sample_weight=w).decision_function(F)
        return S
    return score

def s_ppr(fac, beta, static=False):
    S0 = cho_solve(fac, np.eye(n)[:, q_idx]) if static else None
    def score(pos, neg):
        return S0 if static else cho_solve(fac, pos.astype(float) - beta * neg)
    return score

METHODS = {
    ('cos',        'raw')       : s_cos(X),
    ('cos',        'spread_km') : s_cos(PK),
    ('ppr_static', 'km')        : s_ppr(PPR['km'], 0, static=True),
    ('rocchio',    'raw')       : s_rocchio(X),
    ('rocchio',    'spread')    : s_rocchio(P0),
    ('rocchio',    'spread_km') : s_rocchio(PK),
    ('svm',        'raw')       : s_svm(X),
    ('svm',        'spread')    : s_svm(P0),
    ('svm',        'spread_km') : s_svm(PK),
    ('ppr',        'knn')       : s_ppr(PPR['knn'], 0.5),
    ('ppr',        'km')        : s_ppr(PPR['km'],  0.5),
    ('ppr_pos',    'km')        : s_ppr(PPR['km'],  0.0),     # ignore the not-relevant labels
}

# --
# Simulate

rows = []
for noise in NOISE:
    for (rule, space), score in METHODS.items():
        t0   = time()
        nrng = np.random.default_rng(1000 * seed + int(100 * noise))   # same flips for every method
        pos  = np.zeros((n, Q), dtype=bool)
        pos[q_idx, cols] = True
        neg   = np.zeros((n, Q), dtype=bool)
        found = np.zeros(Q)
        for t in range(T + 1):
            seen = pos | neg
            S    = np.where(seen, -np.inf, score(pos, neg))
            row  = dict(dataset=dataset, seed=seed, noise=noise, rule=rule, space=space, round=t,
                        labels=t * B, ap_unseen=np.nanmean(ap_unseen(S, seen)),
                        recall=(found / n_pos).mean(), eff=(found / np.maximum(np.minimum(t * B, n_pos), 1)).mean())
            if t < T:
                top   = np.argpartition(-S, B, axis=0)[:B]                 # (B, Q)
                true  = rel[top, cols[None,:]]
                obs   = true ^ (nrng.random((B, Q)) < noise)               # what the noisy labeler says
                cc    = np.broadcast_to(cols, (B, Q))
                pos[top[obs],  cc[obs]]  = True
                neg[top[~obs], cc[~obs]] = True
                found += true.sum(0)
                row['batch_prec'] = true.mean()
            rows.append(row)

        print(f'{dataset} seed={seed} | noise={noise} {rule:10s} {space:9s} eff@100={rows[-1]["eff"]:.3f} ({time() - t0:.0f}s)', flush=True)

pd.DataFrame(rows).to_csv(out, index=False)
print(f'{dataset} seed={seed} | done -> {out}', flush=True)
