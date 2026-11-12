"""
  lib.py - the method, and the plumbing shared by baselines.py and ours.py
  
  The method:
      knn_graph()        symmetric k-nearest-neighbor graph
      Graph.ppr()        personalized-PageRank smoothing of anything: one-hot labels, or the vectors themselves
      cluster_metric()   k-means -> ridge onto the cluster indicators -> W'W as a metric      (no labels)
  
  Every method in baselines.py / ours.py has the signature
  
      method(ctx, idx, y_lab, **params) -> predicted class for all n points
  
  where `ctx` holds one sample of a dataset (rows L2-normalized) and caches anything label-free, `idx` are the
  labeled rows and `y_lab` their classes (0 .. C-1).
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from time import time
from scipy.linalg import solve, cho_factor, cho_solve
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='The number of unique classes is greater than 50%')   # 1-shot: every label is unique

# --
# The method

def knn_graph(Z, k=10):
    """ symmetric kNN adjacency: i ~ j if either is among the other's k most cosine-similar points """
    sim = Z @ Z.T
    np.fill_diagonal(sim, -np.inf)
    thresh = np.partition(sim, -k, axis=1)[:,-k]
    adj    = sim >= thresh[:,None]
    return (adj | adj.T).astype(float)

def sqrtm_psd(M):
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T

def cluster_metric(X, K=100, lam=0.5, penalty=2e-4, seed=0, targets=None, Y=None):
    """ label-free metric: k-means -> ridge onto cluster indicators -> AGOP (= W'W for a linear model) -> M^(1/2)
        targets / Y override the k-means step with given integer targets / a given indicator matrix """
    n, d = X.shape
    if Y is None:
        c = KMeans(n_clusters=K, n_init=1, max_iter=100, random_state=seed).fit_predict(X) if targets is None else targets
        Y = np.eye(c.max() + 1)[c]
    Xc   = X - X.mean(0)
    W    = solve(Xc.T @ Xc + penalty * n * np.eye(d), Xc.T @ (Y - Y.mean(0)), assume_a='pos').T
    G    = W.T @ W
    return sqrtm_psd((1 - lam) * np.eye(d) + lam * G * d / np.trace(G))

class Graph:
    def __init__(self, adj):
        self.adj  = adj
        A         = adj + np.eye(len(adj))
        dinv      = 1 / np.sqrt(A.sum(1))
        self.A    = A * dinv[:,None] * dinv[None,:]              # D^-1/2 (A + I) D^-1/2
        self._fac = {}
    
    def ppr(self, B, alpha):
        """ (1 - alpha) (I - alpha A)^-1 B : every row of B replaced by a PageRank-weighted average over the graph """
        if alpha not in self._fac:
            self._fac[alpha] = cho_factor(np.eye(len(self.A)) - alpha * self.A)
        return (1 - alpha) * cho_solve(self._fac[alpha], B)

# --
# One sample of a dataset + its label-free graphs (built on first use, shared by every method and shot count)

class Ctx:
    def __init__(self, X, y, seed=0, k=10, K=100, lam=0.5, penalty=2e-4):
        self.X, self.y, self.seed = normalize(X.astype(np.float64)), y, seed
        self.n, self.d = self.X.shape
        self.C         = int(y.max()) + 1
        self.cfg       = dict(k=k, K=K, lam=lam, penalty=penalty)
        self._cache    = {}
    
    def get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]
    
    def graph(self, name):
        """ 'plain': kNN graph on the vectors.  'cmg': kNN graph under the cluster metric. """
        def build():
            if name == 'plain':
                return Graph(knn_graph(self.X, self.cfg['k']))
            L = cluster_metric(self.X, self.cfg['K'], self.cfg['lam'], self.cfg['penalty'], self.seed)
            return Graph(knn_graph(normalize(self.X @ L), self.cfg['k']))
        return self.get(('graph', name), build)
    
    def spread(self, name, alpha):
        return self.get(('spread', name, alpha), lambda: normalize(self.graph(name).ppr(self.X, alpha)))
    
    def onehot(self, idx, y_lab):
        Y = np.zeros((self.n, self.C))
        Y[idx, y_lab] = 1
        return Y

# --
# Sampling

def sample(X, y, n=5_000, min_count=10, seed=0):
    """ random subset of n rows; classes with fewer than min_count rows in it are dropped; labels remapped to 0..C-1 """
    rng  = np.random.default_rng(seed)
    p    = rng.permutation(len(X))[:n]
    X, y = X[p], y[p]
    _, y = np.unique(y, return_inverse=True)
    keep = np.bincount(y)[y] >= min_count
    X, y = X[keep], y[keep]
    _, y = np.unique(y, return_inverse=True)
    return X, y

def draw_labeled(y, shots, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.where(y == c)[0], shots, replace=False) for c in range(y.max() + 1)]))

# --
# Runner: every method x its small grid, on one (dataset, seed)

def grid(fn, **params):
    """ {variant string : (fn, kwargs)} for the cartesian product of params """
    out = {'' : (fn, {})}
    for k, vals in params.items():
        out = {f'{v0},{k}={v}'.strip(',') : (fn, {**kw, k : v}) for v0, (_, kw) in out.items() for v in vals}
    return out

def run(METHODS, kind):
    """ usage: python blog/<kind>.py <dataset> <seed> [--shots 1 5] [--n 5000] [--only m1,m2 --tag extra] [--out ...] """
    import data
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('seed', type=int)
    parser.add_argument('--shots',   type=int, nargs='+', default=[1, 5])
    parser.add_argument('--n',       type=int, default=5_000)
    parser.add_argument('--encoder', type=str, default=None)
    parser.add_argument('--out',     type=str, default=None)
    parser.add_argument('--only',    type=str, default=None)   # comma-separated method names: run just these...
    parser.add_argument('--tag',     type=str, default=None)   # ...and write to results/<kind>_<tag>/ so finished runs are kept
    args = parser.parse_args()
    if args.only:
        METHODS = {m : METHODS[m] for m in args.only.split(',')}
    
    here = os.path.dirname(os.path.abspath(__file__))
    out  = args.out or os.path.join(here, 'results', kind + (f'_{args.tag}' if args.tag else ''), f'{args.dataset}_{args.seed}.csv')
    if os.path.exists(out):
        return
    
    X, y = data.load(args.dataset, args.encoder)
    X, y = sample(X, y, n=args.n, min_count=2 * max(args.shots), seed=args.seed)
    ctx  = Ctx(X, y, seed=args.seed)
    meta = dict(dataset=args.dataset, group=data.REGISTRY[args.dataset]['group'], seed=args.seed, n=ctx.n, n_classes=ctx.C, kind=kind)
    
    rows, t0 = [], time()
    for shots in args.shots:
        idx  = draw_labeled(y, shots, seed=args.seed)
        test = np.setdiff1d(np.arange(ctx.n), idx)
        for method, variants in METHODS.items():
            for variant, (fn, kw) in variants.items():
                t = time()
                try:
                    yhat = np.asarray(fn(ctx, idx, y[idx], **kw))
                    acc  = float((yhat[test] == y[test]).mean())
                except Exception as e:
                    print(f'!! {args.dataset} seed={args.seed} {method} {variant}: {type(e).__name__}: {e}', flush=True)
                    acc = np.nan
                rows.append(dict(**meta, shots=shots, method=method, variant=variant, acc=acc, secs=time() - t))
    
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'[{kind}] {args.dataset} seed={args.seed} | n={ctx.n} classes={ctx.C} | {time() - t0:.0f}s -> {out}', flush=True)
