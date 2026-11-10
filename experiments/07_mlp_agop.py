"""
  07_mlp_agop.py - does a nonlinear predictor give a better AGOP metric than ridge?

  usage (from repo root): python experiments/07_mlp_agop.py <dataset> <seed> <out.csv>

  Same pipeline as 04_ablate.py (targets -> predictor -> AGOP -> graph metric -> re-spread -> classify),
  swapping the ridge model for sklearn's MLPClassifier with one ReLU hidden layer.

  For f(x) = W2' relu(W1' x + b1) + b2 the Jacobian of the logits at x is  W2' diag(m_x) W1'  (m_x = active units),
  so the AGOP has a closed form, no autograd needed:

      AGOP = W1 [ (W2 W2') * (M'M / n) ] W1'          M = (n, H) matrix of activation masks, * = elementwise

  Also logged: W1 W1' on its own (the "neural feature ansatz": a trained net's first-layer Gram matrix
  is roughly proportional to its AGOP), which needs no data pass at all.

  Targets: k-means clusters (no labels), pseudo-labels from the few-shot classifier, and all true labels (ceiling).
  homophily / oracle rows use ALL labels -- diagnostics only.
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
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

from feature_spreading import MODEL
from feature_spreading.run import load_data
from feature_spreading.agop import metric_sqrt, damp_metric, edge_homophily

warnings.filterwarnings('ignore', category=ConvergenceWarning)

dataset, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NPCS = [1, 5]
K, LAM, HIDDEN = 100, 0.5, 256

splits = {}
for npc in NPCS:
    X_, y_, (_, _, _, _, idx_train, idx_valid) = load_data(dataset, MODEL, 5_000, npc, seed)
    if not splits:
        X, y_raw = X_.astype(np.float64), y_
    splits[npc] = (idx_train, idx_valid)

_, y = np.unique(y_raw, return_inverse=True)
n, d = X.shape
C    = y.max() + 1

rows = []
def log(method, npc, pred, scores, **kw):
    idx_valid = splits[npc][1]
    rows.append(dict(dataset=dataset, seed=seed, npc=npc, method=method, pred=pred,
                     acc=(scores.argmax(1)[idx_valid] == y[idx_valid]).mean(), **kw))

# --
# Building blocks (same as 04_ablate.py)

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

def regraph(G):
    adj = knn_graph(normalize(X @ metric_sqrt(damp_metric(G, LAM))))
    return adj, normalize(spread(adj, X))

def evaluate(method, adj, P, npcs=NPCS, **cfg):
    hom = edge_homophily(adj, y)
    Y   = np.zeros((n, C * len(npcs)))
    for i, npc in enumerate(npcs):
        Y[splits[npc][0], i * C + y[splits[npc][0]]] = 1
    F = spread(adj, Y, 0.95)
    for i, npc in enumerate(npcs):
        log(method, npc, 'svc', fit_svc(P, npc),          homophily=hom, **cfg)
        log(method, npc, 'lp',  F[:, i * C:(i + 1) * C],  homophily=hom, **cfg)

# --
# Predictors -> AGOP

def agop_ridge(F, lab, reg=1.0):
    m  = lab.max() + 1
    Y  = -np.ones((n, m))
    Y[np.arange(n), lab] = 1
    Fc = F - F.mean(0)
    W  = solve(Fc.T @ Fc + reg * np.eye(d), Fc.T @ (Y - Y.mean(0)), assume_a='pos').T
    return W.T @ W

def agop_mlp(F, lab, alpha):
    # inputs are unit-norm rows (entries ~ 1/sqrt(d)); rescale so Adam's default step size is sensible
    Fs  = F * np.sqrt(d)
    mlp = MLPClassifier(hidden_layer_sizes=(HIDDEN,), alpha=alpha, max_iter=200, random_state=seed).fit(Fs, lab)
    W1, W2 = mlp.coefs_                                       # (d, H), (H, n_out)
    M      = (Fs @ W1 + mlp.intercepts_[0] > 0).astype(float) # (n, H) activation masks
    G_agop = W1 @ ((W2 @ W2.T) * (M.T @ M / n)) @ W1.T
    G_w1   = W1 @ W1.T
    return G_agop, G_w1, dict(train_acc=mlp.score(Fs, lab), n_iter=mlp.n_iter_, mlp_alpha=alpha)

# --
# Run

t0   = time()
adj0 = knn_graph(X)
P0   = normalize(spread(adj0, X))
evaluate('ppr', adj0, P0)

scores0 = {npc : fit_svc(P0, npc) for npc in NPCS}

def targets():
    yield 'kmeans', NPCS, KMeans(n_clusters=K, n_init=1, max_iter=100, random_state=seed).fit(P0).labels_
    for npc in NPCS:
        lab = scores0[npc].argmax(1)
        lab[splits[npc][0]] = y[splits[npc][0]]
        yield 'pseudo', [npc], lab
    yield 'oracle', NPCS, y

for tname, npcs, lab in targets():
    adj, P = regraph(agop_ridge(P0, lab))
    evaluate('ridge', adj, P, npcs=npcs, targets=tname)

    for alpha in [1e-4, 1.0, 10.0, 100.0]:
        G_agop, G_w1, info = agop_mlp(P0, lab, alpha)
        adj, P = regraph(G_agop)
        evaluate('mlp_agop', adj, P, npcs=npcs, targets=tname, **info)
        adj, P = regraph(G_w1)
        evaluate('mlp_w1', adj, P, npcs=npcs, targets=tname, **info)

pd.DataFrame(rows).to_csv(out, index=False)
print(f'{dataset} seed={seed} | done -> {out} ({time() - t0:.0f}s)', flush=True)
