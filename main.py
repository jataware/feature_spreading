#!/usr/env/python

"""
  main.py
"""

import os
import argparse
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from sklearn.svm import LinearSVC
from sklearn.semi_supervised import LabelPropagation

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
  parser.add_argument('--n_obs',        type=int, default=5_000)
  parser.add_argument('--n_per_class',  type=int, default=1)
  parser.add_argument('--seed',         type=int, default=234)
  args = parser.parse_args()
  
  assert args.dataset in _DATASETS
  
  return args


args = parse_args()
np.random.seed(args.seed)

# --
# IO / ETL

X = np.load(os.path.join('/home/ubuntu/data/img_feats', args.dataset, 'train', args.model, 'X.npy'))
y = np.load(os.path.join('/home/ubuntu/data/img_feats', args.dataset, 'train', args.model, 'y.npy'))

X, y = get_sample(X, y, args.n_obs)

X = normalize(X)

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
# Baseline: Label Propagation

y_ss = y.copy()
y_ss[idx_valid] = -1

# t       = time()
# lp      = LabelPropagation(kernel='knn', n_neighbors=10, n_jobs=-1, max_iter=2_000).fit(X, y_ss)
# lp_acc  = (y_valid == lp.transduction_[idx_valid]).mean()
# lp_time = time() - t
# print(f'[baseline] lp  {lp_acc:0.5f} {lp_time:0.5f}')

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

def ppr_smoothing(X, alpha, k, do_normalize=True):
    adj = compute_knn_graph(X, k=k)     # compute mutual KNN graph
    ppr = compute_ppr(adj, alpha=alpha) # compute PPR matrix
    P   = ppr @ X
    
    if do_normalize:
        P = normalize(P, axis=1)
    
    return P


t = time()

P = ppr_smoothing(X, k=10, alpha=0.85)
P_train, P_valid = P[idx_train], P[idx_valid]

svc       = LinearSVC().fit(P_train, y_train) # train model on PPR-average features
ppr_preds = svc.predict(P_valid)

ppr_acc   = (y_valid == ppr_preds).mean()
ppr_f1    = f1_score(y_valid, ppr_preds, average='macro')
ppr_time  = time() - t

print(f'[  ours  ] ppr {ppr_acc:0.5f} {ppr_f1:0.5f} {ppr_time:0.5f}')
