"""
  baselines.py - baselines, each one a call into an existing library (scikit-learn or graphlearning)
  
  usage (from repo root): python blog/baselines.py <dataset> <seed>
  
  scikit-learn, ignoring the unlabeled points:
      svm                 LinearSVC
      logreg              LogisticRegression
      nearest_centroid    NearestCentroid
      knn                 KNeighborsClassifier(1)
  scikit-learn, semi-supervised:
      label_propagation   LabelPropagation          (rbf and knn kernels)
      label_spreading     LabelSpreading            (rbf and knn kernels)
      self_training       SelfTrainingClassifier around LogisticRegression
      kmeans_seeded       KMeans with one centroid per class, started at the labeled class means
  graphlearning (Calder), on the kNN graph that library builds:
      laplace             Laplace learning / harmonic functions (Zhu et al. 2003)
      poisson             Poisson learning (Calder et al. 2020), designed for very few labels
      poisson_connected   the same, with the neighbor count raised until the graph is connected
      randomwalk, graph_nn, centered_kernel
      (left out: poisson_mbo needs the true class proportions, which no other method is given;
       sparse_label_propagation scored 7% on a first test and takes far longer than anything else here)
  
  No method here trains a network. Hyperparameters: library defaults, plus the small grids below.
"""

import os
import sys
import warnings
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.neighbors')   # NearestCentroid at 1-shot: 0 / 0 in an unused variance

# -- scikit-learn, supervised

def svm(ctx, idx, y_lab, C=1.0):
    return LinearSVC(C=C).fit(ctx.X[idx], y_lab).predict(ctx.X)

def logreg(ctx, idx, y_lab, C=1.0):
    return LogisticRegression(C=C, max_iter=500).fit(ctx.X[idx], y_lab).predict(ctx.X)

def nearest_centroid(ctx, idx, y_lab):
    return NearestCentroid().fit(ctx.X[idx], y_lab).predict(ctx.X)

def knn(ctx, idx, y_lab):
    return KNeighborsClassifier(n_neighbors=1).fit(ctx.X[idx], y_lab).predict(ctx.X)

# -- scikit-learn, semi-supervised

def _y_ss(ctx, idx, y_lab):
    y = -np.ones(ctx.n, dtype=int)
    y[idx] = y_lab
    return y

def label_propagation(ctx, idx, y_lab, kernel='rbf'):
    # defaults, except: 10 neighbors for the knn kernel, and an iteration cap for the (dense, slow) rbf kernel
    m = LabelPropagation(kernel=kernel, n_neighbors=10, max_iter=1000 if kernel == 'knn' else 100)
    return m.fit(ctx.X, _y_ss(ctx, idx, y_lab)).transduction_

def label_spreading(ctx, idx, y_lab, kernel='rbf', alpha=0.2):
    m = LabelSpreading(kernel=kernel, n_neighbors=10, alpha=alpha, max_iter=200)
    return m.fit(ctx.X, _y_ss(ctx, idx, y_lab)).transduction_

def self_training(ctx, idx, y_lab, C=10.0):
    m = SelfTrainingClassifier(LogisticRegression(C=C, max_iter=200), criterion='k_best', k_best=ctx.n // 5, max_iter=5)
    return m.fit(ctx.X, _y_ss(ctx, idx, y_lab)).predict(ctx.X)

def kmeans_seeded(ctx, idx, y_lab):
    init = np.stack([ctx.X[idx][y_lab == c].mean(0) for c in range(ctx.C)])
    return KMeans(n_clusters=ctx.C, init=init, n_init=1, max_iter=100).fit_predict(ctx.X)

# -- graphlearning

def _gl(name):
    def method(ctx, idx, y_lab):
        import graphlearning as gl
        def build():   # the library's own graph (Gaussian weights, symmetrized), with its exact neighbor search
            return gl.weightmatrix.knn(ctx.X, 10, knn_data=gl.weightmatrix.knnsearch(ctx.X, 10, method='brute'))
        W     = ctx.get('gl_W', build)
        model = getattr(gl.ssl, name)(W)
        return model.fit_predict(np.asarray(idx), np.asarray(y_lab))
    return method

def poisson_connected(ctx, idx, y_lab):
    """ Poisson learning needs a connected graph: it collapses to 2-3 predicted classes when the kNN graph has even one
        small extra component. This raises the neighbor count until the library's graph is connected. """
    import graphlearning as gl
    from scipy.sparse.csgraph import connected_components
    def build():
        for k in [10, 15, 20, 30, 50, 100]:
            W = gl.weightmatrix.knn(ctx.X, k, knn_data=gl.weightmatrix.knnsearch(ctx.X, k, method='brute'))
            if connected_components(W, directed=False)[0] == 1:
                break
        return W
    W = ctx.get('gl_W_connected', build)
    return gl.ssl.poisson(W).fit_predict(np.asarray(idx), np.asarray(y_lab))

METHODS = {
    'svm'               : lib.grid(svm, C=[1, 10, 100]),
    'logreg'            : lib.grid(logreg, C=[1, 10, 100, 1000]),
    'nearest_centroid'  : lib.grid(nearest_centroid),
    'knn'               : lib.grid(knn),
    'label_propagation' : lib.grid(label_propagation, kernel=['rbf', 'knn']),
    'label_spreading'   : lib.grid(label_spreading, kernel=['rbf', 'knn'], alpha=[0.2, 0.8]),
    'self_training'     : lib.grid(self_training, C=[10, 100]),
    'kmeans_seeded'     : lib.grid(kmeans_seeded),
    'laplace'           : lib.grid(_gl('laplace')),
    'poisson'           : lib.grid(_gl('poisson')),
    'poisson_connected' : lib.grid(poisson_connected),
    'randomwalk'        : lib.grid(_gl('randomwalk')),
    'graph_nn'          : lib.grid(_gl('graph_nearest_neighbor')),
    'centered_kernel'   : lib.grid(_gl('centered_kernel')),
}

if __name__ == '__main__':
    lib.run(METHODS, 'baselines')
