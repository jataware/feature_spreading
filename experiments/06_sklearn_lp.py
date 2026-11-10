"""
  06_sklearn_lp.py - why does sklearn's label propagation "not work" here, when PPR over the repo's graph does?
  
  usage (from repo root): python experiments/06_sklearn_lp.py <dataset> <seed> <out.csv>
  
  sklearn's kernel='knn' graph is *directed*: row i has a 1 for each of i's nearest neighbors, and the update
  F <- S F means point i only ever pulls labels from its own neighbor list. A labeled point that is in nobody
  else's neighbor list can never pass its label on. This script tests that explanation by swapping in the
  symmetrized graph (i ~ j if either lists the other) and changing nothing else.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import solve
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from feature_spreading import MODEL
from feature_spreading.run import load_data, compute_knn_graph

warnings.filterwarnings('ignore', category=ConvergenceWarning)

dataset, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]

rows = []
for npc in [1, 5]:
    X, y, (_, _, _, _, idx_train, idx_valid) = load_data(dataset, MODEL, 5_000, npc, seed)
    X    = X.astype(np.float64)
    _, y = np.unique(y, return_inverse=True)
    n, C = X.shape[0], y.max() + 1
    y_ss = -np.ones(n, dtype=int)
    y_ss[idx_train] = y[idx_train]
    
    def log(method, yhat, **kw):
        rows.append(dict(dataset=dataset, seed=seed, npc=npc, method=method,
                         acc=(yhat[idx_valid] == y[idx_valid]).mean(), **kw))
    
    # sklearn's directed graph, exactly as LabelSpreading builds it
    A_dir = NearestNeighbors(n_neighbors=10).fit(X).kneighbors_graph(X, 10, mode='connectivity')
    A_dir = sparse.csr_matrix(A_dir)
    A_dir.setdiag(0)
    A_dir.eliminate_zeros()
    in_deg = np.asarray(A_dir.sum(0)).ravel()                 # how many points list i as a neighbor
    diag   = dict(
        frac_labeled_never_listed = (in_deg[idx_train] == 0).mean(),
        frac_all_never_listed     = (in_deg == 0).mean(),
    )
    
    # symmetrized versions of the same graph
    A_sym    = ((A_dir + A_dir.T) > 0).astype(float)
    sym_sp   = lambda *_: sparse.coo_matrix(A_sym)
    sym_den  = lambda *_: A_sym.toarray()
    
    # 1. sklearn as shipped (the repo's original, commented-out baseline)
    m = LabelPropagation(kernel='knn', n_neighbors=10, max_iter=2_000).fit(X, y_ss)
    log('sklearn LabelPropagation (directed kNN)', m.transduction_, n_iter=m.n_iter_,
        frac_unreached=(m.label_distributions_.sum(1) == 0).mean(), **diag)
    
    m = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.95, max_iter=2_000).fit(X, y_ss)
    log('sklearn LabelSpreading (directed kNN)', m.transduction_, n_iter=m.n_iter_,
        frac_unreached=(m.label_distributions_.sum(1) == 0).mean(), **diag)
    
    # 2. sklearn's algorithms, symmetrized graph, nothing else changed
    m = LabelPropagation(kernel=sym_den, max_iter=500).fit(X, y_ss)
    log('sklearn LabelPropagation (symmetrized kNN)', m.transduction_, n_iter=m.n_iter_,
        frac_unreached=(m.label_distributions_.sum(1) == 0).mean())
    
    m = LabelSpreading(kernel=sym_sp, alpha=0.95, max_iter=2_000).fit(X, y_ss)
    log('sklearn LabelSpreading (symmetrized kNN)', m.transduction_, n_iter=m.n_iter_,
        frac_unreached=(m.label_distributions_.sum(1) == 0).mean())
    
    # 3. this repo: symmetrized graph + self-loops, closed-form PPR on one-hot labels
    adj  = compute_knn_graph(X, k=10)
    A    = adj + np.eye(n)
    dinv = 1 / np.sqrt(A.sum(1))
    Y    = np.zeros((n, C))
    Y[idx_train, y[idx_train]] = 1
    F    = solve(np.eye(n) - 0.95 * (A * dinv[:,None] * dinv[None,:]), Y, assume_a='pos')
    log('PPR on one-hot labels (repo graph)', F.argmax(1))

pd.DataFrame(rows).to_csv(out, index=False)
print(f'{dataset} seed={seed} | done -> {out}', flush=True)
