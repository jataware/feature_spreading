#!/usr/bin/env python

"""
  main_agop.py

  main.py + "recursive feature spreading":
    graph (metric M) -> PPR smooth -> fit -> AGOP -> M -> repeat
"""

import os
import argparse
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from sklearn.svm import LinearSVC

_DATASETS = [
  # "cifar10",
  "Multimodal-Fatima/OxfordPets_train",
  "Multimodal-Fatima/StanfordCars_train",
  "jonathan-roberts1/NWPU-RESISC45",
  "nelorth/oxford-flowers",
  "fashion_mnist",
  "food101",
]

# --
# Helpers

def get_sample(X, y, n_obs):
    # shuffle
    p    = np.random.permutation(X.shape[0])
    X, y = X[p], y[p]

    # subset
    X, y = X[:n_obs], y[:n_obs]

    # contiguous
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)

    return X, y

def balanced_train_test_split(X, y, n_per_class):
    idxs      = np.arange(X.shape[0])
    idx_train = pd.Series(idxs).groupby(y).apply(lambda x: x.sample(n_per_class)).values
    idx_valid = np.setdiff1d(idxs, idx_train)

    return X[idx_train], X[idx_valid], y[idx_train], y[idx_valid], idx_train, idx_valid

# --
# IO

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset',      type=str, default='jonathan-roberts1/NWPU-RESISC45')
  parser.add_argument('--model',        type=str, default='openai/clip-vit-large-patch14')
  parser.add_argument('--data_dir',     type=str, default='/home/ubuntu/data/img_feats')
  parser.add_argument('--n_obs',        type=int, default=5_000)
  parser.add_argument('--n_per_class',  type=int, default=1)
  parser.add_argument('--seed',         type=int, default=234)

  # AGOP extension
  parser.add_argument('--agop',         type=str,   default='kernel', choices=['kernel', 'linear'])
  parser.add_argument('--n_iters',      type=int,   default=4)
  parser.add_argument('--lam',          type=float, default=0.5)  # 0 = cosine metric, 1 = pure AGOP metric
  parser.add_argument('--krr_reg',      type=float, default=1e-3)
  args = parser.parse_args()

  assert args.dataset in _DATASETS

  return args


args = parse_args()
np.random.seed(args.seed)

# --
# IO / ETL

X = np.load(os.path.join(args.data_dir, args.dataset, 'train', args.model, 'X.npy'))
y = np.load(os.path.join(args.data_dir, args.dataset, 'train', args.model, 'y.npy'))

X, y = get_sample(X, y, args.n_obs)

X = normalize(X).astype(np.float64)

X_train, X_valid, y_train, y_valid, idx_train, idx_valid =\
    balanced_train_test_split(X, y, args.n_per_class)

# --
# Baseline: SVC

t         = time()
svc       = LinearSVC().fit(X_train, y_train)
svc_preds = svc.predict(X_valid)
svc_acc   = (y_valid == svc_preds).mean()
svc_f1    = f1_score(y_valid, svc_preds, average='macro')
svc_time  = time() - t

print(f'[baseline] svc {svc_acc:0.5f} {svc_f1:0.5f} {svc_time:0.5f}')

# --
# Our Method: "PPR Feature Spreading"
# !! Inefficient implementation ... should use sparse matrices

def calc_A_hat(adj):
    A     = adj + np.eye(adj.shape[0])
    D     = np.sum(A, axis=1)
    D_inv = np.diag(1 / np.sqrt(D))
    return D_inv @ A @ D_inv

def compute_ppr(adj, alpha):
    ralpha  = 1 - alpha
    A_hat   = calc_A_hat(adj)
    A_inner = np.eye(adj.shape[0]) - (1 - ralpha) * A_hat
    return ralpha * np.linalg.inv(A_inner)

def compute_knn_graph(X, k=10):
    sim     = X @ X.T
    thresh  = np.partition(sim, -k, axis=-1)[:,-k] # kth largest entry
    adj     = (sim > thresh[:,None])
    adj     = (adj | adj.T).astype(float)
    return adj

# --
# Extension: learn the metric with AGOP, rebuild the graph, iterate
#
#   cosine sim  x'z / |x||z|   ->   x'Mz / |x|_M |z|_M
#   (== plain cosine sim after mapping x -> M^{1/2} x)

def metric_sqrt(M):
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T

def damp_metric(M, lam):
    # trace-normalize so M is comparable to I, then shrink toward I
    d = M.shape[0]
    return (1 - lam) * np.eye(d) + lam * M * (d / np.trace(M))

def sqdist(A, B):
    d2 = (A ** 2).sum(1)[:,None] + (B ** 2).sum(1)[None,:] - 2 * A @ B.T
    return np.clip(d2, 0, None)

def agop_linear(svc):
    # linear head: f(x) = Wx  =>  grad is W everywhere  =>  AGOP = W'W
    W = svc.coef_
    return W.T @ W

def agop_kernel(P, P_train, y_train, M, reg=1e-3, bs=256):
    # RFM step: gaussian kernel ridge under metric M, fit on the *labeled* points only,
    # but gradients are averaged over *all* points -- the unlabeled data tells us
    # where the predictor's sensitivity actually matters.
    L     = metric_sqrt(M)
    Q, Z  = P @ L, P_train @ L
    D2    = sqdist(Q, Z)
    s2    = np.median(D2)                                   # bandwidth: median heuristic

    Y     = (y_train[:,None] == np.unique(y_train)[None,:]).astype(float)
    K_zz  = np.exp(-sqdist(Z, Z) / (2 * s2))
    coef  = np.linalg.solve(K_zz + reg * np.eye(Z.shape[0]), Y)  # (n_lab, n_class)

    G     = np.exp(-D2 / (2 * s2))                          # (n, n_lab)
    PM    = P @ M
    ZM    = P_train @ M

    d     = P.shape[1]
    agop  = np.zeros((d, d))
    for i in range(0, P.shape[0], bs):
        g = G[i:i + bs]
        # J[x] = d f(x) / d x  (n_class, d), up to the constant -1/s2
        J = (g @ coef)[:,:,None] * PM[i:i + bs][:,None,:] - np.einsum('bi,ic,id->bcd', g, coef, ZM)
        J = J.reshape(-1, d)
        agop += J.T @ J

    return agop / P.shape[0]


M = np.eye(X.shape[1])
for it in range(args.n_iters + 1):
    t = time()

    X_m = normalize(X @ metric_sqrt(M)) if it > 0 else X    # metric only changes the graph
    adj = compute_knn_graph(X_m, k=10)
    ppr = compute_ppr(adj, alpha=0.85)
    P   = normalize(ppr @ X, axis=1)                        # still spreading the original features
    P_train, P_valid = P[idx_train], P[idx_valid]

    svc       = LinearSVC().fit(P_train, y_train)
    ppr_preds = svc.predict(P_valid)

    ppr_acc   = (y_valid == ppr_preds).mean()
    ppr_f1    = f1_score(y_valid, ppr_preds, average='macro')

    # graph quality, independent of the classifier: fraction of edges joining same-class nodes
    adj_od    = adj * (1 - np.eye(adj.shape[0]))             # drop self-loops
    homoph    = (adj_od * (y[:,None] == y[None,:])).sum() / adj_od.sum()

    # AGOP -> next metric  (labels used: y_train only)
    if args.agop == 'linear':
        G_ = agop_linear(svc)
    else:
        G_ = agop_kernel(P, P_train, y_train, M, reg=args.krr_reg)

    M = damp_metric(G_, args.lam)

    tag = '[  ours  ] ppr' if it == 0 else f'[agop it{it}] ppr'
    print(f'{tag} {ppr_acc:0.5f} {ppr_f1:0.5f} {time() - t:0.5f} | edge_homophily {homoph:0.5f}')
