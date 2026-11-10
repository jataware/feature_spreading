"""
  agop.py - "recursive feature spreading"
  
    graph (metric M) -> PPR smooth -> fit -> AGOP -> M -> repeat
  
  cosine sim  x'z / |x||z|   ->   x'Mz / |x|_M |z|_M
  (== plain cosine sim after mapping x -> M^{1/2} x)
"""

import numpy as np
from time import time
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from .run import load_data, compute_knn_graph, compute_ppr
from .pretty import print_results

# --
# Helpers

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

def edge_homophily(adj, y):
    # graph quality, independent of the classifier: fraction of edges joining same-class nodes
    # !! uses all labels -- diagnostic only, never feeds back into the method
    adj = adj * (1 - np.eye(adj.shape[0])) # drop self-loops
    return (adj * (y[:,None] == y[None,:])).sum() / adj.sum()

# --
# AGOP

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
    s2    = np.median(D2)                                        # bandwidth: median heuristic
    
    Y     = (y_train[:,None] == np.unique(y_train)[None,:]).astype(float)
    K_zz  = np.exp(-sqdist(Z, Z) / (2 * s2))
    coef  = np.linalg.solve(K_zz + reg * np.eye(Z.shape[0]), Y)  # (n_lab, n_class)
    
    G     = np.exp(-D2 / (2 * s2))                               # (n, n_lab)
    PM    = P @ M
    ZM    = P_train @ M
    
    d     = P.shape[1]
    agop  = np.zeros((d, d))
    for i in range(0, P.shape[0], bs):
        g = G[i:i + bs]
        # J[x] = d f(x) / d x  (n_class, d), up to the constant -1/s2
        #   = (sum_i g_i coef_ic) Mx - sum_i g_i coef_ic M z_i
        J = (g @ coef)[:,:,None] * PM[i:i + bs][:,None,:] - (g[:,:,None] * coef[None]).transpose(0, 2, 1) @ ZM
        J = J.reshape(-1, d)
        agop += J.T @ J
    
    return agop / P.shape[0]

# --
# Run

def run_agop(dataset, model, n_obs, n_per_class, seed, agop, n_iters, lam, krr_reg):
    X, y, (X_train, X_valid, y_train, y_valid, idx_train, idx_valid) = \
        load_data(dataset, model, n_obs, n_per_class, seed)
    
    X = X.astype(np.float64)
    
    results = {}
    
    # --
    # Baseline: SVC
    
    t         = time()
    svc       = LinearSVC().fit(X_train, y_train)
    svc_preds = svc.predict(X_valid)
    results['svc (baseline)'] = {
        'acc'  : (y_valid == svc_preds).mean(),
        'f1'   : f1_score(y_valid, svc_preds, average='macro'),
        'time' : time() - t,
    }
    
    # --
    # PPR feature spreading, w/ graph metric learned by AGOP
    
    M = np.eye(X.shape[1])
    for it in range(n_iters + 1):
        t = time()
        
        X_m = normalize(X @ metric_sqrt(M)) if it > 0 else X # metric only changes the graph
        adj = compute_knn_graph(X_m, k=10)
        ppr = compute_ppr(adj, alpha=0.85)
        P   = normalize(ppr @ X, axis=1)                     # still spreading the original features
        P_train, P_valid = P[idx_train], P[idx_valid]
        
        svc       = LinearSVC().fit(P_train, y_train)
        ppr_preds = svc.predict(P_valid)
        
        name = 'ppr (ours)' if it == 0 else f'ppr + agop it{it}'
        results[name] = {
            'acc'       : (y_valid == ppr_preds).mean(),
            'f1'        : f1_score(y_valid, ppr_preds, average='macro'),
            'homophily' : edge_homophily(adj, y),
        }
        
        # AGOP -> next metric  (labels used: y_train only)
        if it < n_iters:
            if agop == 'linear':
                G_ = agop_linear(svc)
            else:
                G_ = agop_kernel(P, P_train, y_train, M, reg=krr_reg)
            
            M = damp_metric(G_, lam)
        
        results[name]['time'] = time() - t # includes AGOP step
    
    print_results(f'{dataset} | n_obs={X.shape[0]} n_per_class={n_per_class} | agop={agop} lam={lam}', results)
