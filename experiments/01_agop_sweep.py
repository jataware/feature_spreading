"""
  sweep.py - grid over AGOP params + oracle diagnostics
  usage: pixi run python sweep.py <dataset> <out.csv> [n_per_class ...]
"""

import sys
import numpy as np
import pandas as pd
from time import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

from feature_spreading import MODEL
from feature_spreading.run import load_data, compute_knn_graph, compute_ppr
from feature_spreading.agop import metric_sqrt, damp_metric, agop_linear, agop_kernel, edge_homophily

dataset, out = sys.argv[1], sys.argv[2]
NPCS    = [int(x) for x in sys.argv[3:]] or [1]
SEEDS   = [0, 1, 2]
LAMS    = [0.25, 0.5, 0.75, 1.0]
REGS    = [1e-3, 1e-1]
N_ITERS = 4

def eff_rank(G):
    return np.trace(G) ** 2 / (G ** 2).sum()

def spread(X, y, adj, idx_train, idx_valid):
    P   = normalize(compute_ppr(adj, alpha=0.85) @ X)
    svc = LinearSVC().fit(P[idx_train], y[idx_train])
    acc = (svc.predict(P[idx_valid]) == y[idx_valid]).mean()
    return P, svc, acc

def graph(X, M):
    return compute_knn_graph(normalize(X @ metric_sqrt(M)), k=10)

rows = []
def log(**kw):
    rows.append(dict(dataset=dataset, **kw))
    pd.DataFrame(rows).to_csv(out, index=False)

for npc in NPCS:
    for seed in SEEDS:
        X, y, (_, _, _, _, idx_train, idx_valid) = load_data(dataset, MODEL, 5_000, npc, seed)
        X     = X.astype(np.float64)
        y_tr  = y[idx_train]
        base  = dict(npc=npc, seed=seed)

        # --
        # base ppr

        adj0          = compute_knn_graph(X, k=10)
        P0, svc0, acc0 = spread(X, y, adj0, idx_train, idx_valid)
        log(**base, method='ppr', it=0, acc=acc0, homophily=edge_homophily(adj0, y), overlap=1.0)
        print(f'{dataset} npc={npc} seed={seed} ppr acc={acc0:.4f}', flush=True)

        # --
        # oracles (use ALL labels -- headroom diagnostics only)

        # perfect graph: drop every cross-class edge from the base graph
        adj_o = adj0 * (y[:,None] == y[None,:])
        _, _, acc = spread(X, y, adj_o, idx_train, idx_valid)
        log(**base, method='oracle_graph', it=1, acc=acc, homophily=1.0, overlap=adj_o.sum() / adj0.sum())

        # perfect classifier: linear AGOP from an SVC fit on all labels
        G_o = agop_linear(LinearSVC().fit(P0, y))
        for lam in [0.5, 1.0]:
            adj = graph(X, damp_metric(G_o, lam))
            _, _, acc = spread(X, y, adj, idx_train, idx_valid)
            log(**base, method='oracle_agop_linear', lam=lam, it=1, acc=acc,
                homophily=edge_homophily(adj, y), overlap=(adj * adj0).sum() / adj0.sum(), eff_rank=eff_rank(G_o))

        # --
        # grid

        configs = [('kernel', lam, reg) for lam in LAMS for reg in REGS] + [('linear', lam, None) for lam in LAMS]
        for agop, lam, reg in configs:
            t = time()
            P, svc = P0, svc0
            M      = np.eye(X.shape[1])
            for it in range(1, N_ITERS + 1):
                G_  = agop_linear(svc) if agop == 'linear' else agop_kernel(P, P[idx_train], y_tr, M, reg=reg)
                M   = damp_metric(G_, lam)
                adj = graph(X, M)
                P, svc, acc = spread(X, y, adj, idx_train, idx_valid)
                log(**base, method=f'agop_{agop}', lam=lam, reg=reg, it=it, acc=acc,
                    homophily=edge_homophily(adj, y), overlap=(adj * adj0).sum() / adj0.sum(), eff_rank=eff_rank(G_))

            print(f'  {agop:6s} lam={lam} reg={reg} acc={acc:.4f} ({time() - t:.1f}s)', flush=True)
