"""
  ours.py - our methods: personalized-PageRank smoothing over a symmetric kNN graph, plain or cluster-metric
  
  usage (from repo root): python blog/ours.py <dataset> <seed>
  
      ls        label spreading:    spread the one-hot labels over the graph, take the largest
      fs        feature spreading:  spread the vectors over the graph, train a linear SVM on the labeled rows
      fs+ls     both: add the two methods' standardized class scores
  
      cmg_*     the same three, on the cluster-metric graph (k-means -> ridge -> W'W metric -> second kNN graph).
                LABEL-FREE: the graph is built once and shared by every labeled set.
      plg_*     the same three, on a graph whose metric is learned from the clusters AND pseudo-labels (the plain-graph
                classifier's predictions for every point). USES THE LABELS to build the graph, so it is rebuilt for
                each labeled set. Reported separately from the label-free graphs.
  
      oracle_*  ceilings, not methods: label spreading on a graph built with every true label
"""

import os
import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

def pseudo_label_graph(ctx, idx, y_lab):
    """ metric from [pseudo-labels | k-means clusters] -> kNN graph. Label-dependent, so cached per labeled set. """
    def build():
        P      = ctx.spread('plain', 0.85)
        pseudo = LinearSVC().fit(P[idx], y_lab).predict(P)
        pseudo[idx] = y_lab
        K = ctx.cfg['K']
        c = ctx.get('kmeans', lambda: KMeans(n_clusters=K, n_init=1, max_iter=100, random_state=ctx.seed).fit_predict(ctx.X))
        Y = np.column_stack([np.eye(ctx.C)[pseudo], np.eye(K)[c]])
        L = lib.cluster_metric(ctx.X, lam=ctx.cfg['lam'], penalty=ctx.cfg['penalty'], Y=Y)
        return lib.Graph(lib.knn_graph(normalize(ctx.X @ L), ctx.cfg['k']))
    return ctx.get(('plg', idx.tobytes()), build)

def get_graph(ctx, graph, idx, y_lab):
    return pseudo_label_graph(ctx, idx, y_lab) if graph == 'plg' else ctx.graph(graph)

def spread_vectors(ctx, graph, idx, y_lab, alpha):
    if graph != 'plg':
        return ctx.spread(graph, alpha)
    g = pseudo_label_graph(ctx, idx, y_lab)
    return ctx.get(('plg_spread', idx.tobytes(), alpha), lambda: normalize(g.ppr(ctx.X, alpha)))

def ls_scores(ctx, idx, y_lab, graph, alpha):
    return get_graph(ctx, graph, idx, y_lab).ppr(ctx.onehot(idx, y_lab), alpha)

def fs_scores(ctx, idx, y_lab, graph, alpha):
    P = spread_vectors(ctx, graph, idx, y_lab, alpha)
    S = LinearSVC().fit(P[idx], y_lab).decision_function(P)
    return S if S.ndim == 2 else np.column_stack([-S, S])

def zscore(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)

def ls(ctx, idx, y_lab, graph='plain', alpha=0.95):
    return ls_scores(ctx, idx, y_lab, graph, alpha).argmax(1)

def fs(ctx, idx, y_lab, graph='plain', alpha=0.85):
    return fs_scores(ctx, idx, y_lab, graph, alpha).argmax(1)

def fs_ls(ctx, idx, y_lab, graph='plain', alpha_ls=0.95, alpha_fs=0.85):
    return (zscore(ls_scores(ctx, idx, y_lab, graph, alpha_ls)) + zscore(fs_scores(ctx, idx, y_lab, graph, alpha_fs))).argmax(1)

def oracle(ctx, idx, y_lab, which='graph', alpha=0.95):
    def build():
        if which == 'graph':    # the plain graph with every cross-class edge removed
            return lib.Graph(ctx.graph('plain').adj * (ctx.y[:,None] == ctx.y[None,:]))
        L = lib.cluster_metric(ctx.X, lam=ctx.cfg['lam'], penalty=ctx.cfg['penalty'], targets=ctx.y)   # metric from the true classes
        return lib.Graph(lib.knn_graph(normalize(ctx.X @ L), ctx.cfg['k']))
    g = ctx.get(('oracle', which), build)
    return g.ppr(ctx.onehot(idx, y_lab), alpha).argmax(1)

A_LS, A_FS = [0.9, 0.95, 0.99], [0.5, 0.85]

METHODS = {
    'ls'            : lib.grid(ls,    graph=['plain'], alpha=A_LS),
    'fs'            : lib.grid(fs,    graph=['plain'], alpha=A_FS),
    'fs+ls'         : lib.grid(fs_ls, graph=['plain'], alpha_ls=A_LS, alpha_fs=A_FS),
    'cmg_ls'        : lib.grid(ls,    graph=['cmg'],   alpha=A_LS),
    'cmg_fs'        : lib.grid(fs,    graph=['cmg'],   alpha=A_FS),
    'cmg_fs+ls'     : lib.grid(fs_ls, graph=['cmg'],   alpha_ls=A_LS, alpha_fs=A_FS),
    'plg_ls'        : lib.grid(ls,    graph=['plg'],   alpha=A_LS),
    'plg_fs'        : lib.grid(fs,    graph=['plg'],   alpha=A_FS),
    'plg_fs+ls'     : lib.grid(fs_ls, graph=['plg'],   alpha_ls=A_LS, alpha_fs=A_FS),
    'oracle_graph'  : lib.grid(oracle, which=['graph']),
    'oracle_metric' : lib.grid(oracle, which=['metric']),
}

if __name__ == '__main__':
    lib.run(METHODS, 'ours')
