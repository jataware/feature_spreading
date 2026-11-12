"""
  make_report.py - compile experiments/results/* into a standalone experiments/report.html

  usage (from repo root): python experiments/make_report.py

  Every number in the report is read from the result CSVs at build time. No network, no dependencies
  in the output: one HTML file with inline CSS, inline SVG charts and a 30-line tooltip script.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_lib import esc, f3, sgn, legend, dotplot, linepanel, divbars, toy, table, best_cells

HERE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{HERE}/results/{name}/*.csv'))], ignore_index=True)

A  = load('04_ablate')
R  = load('05_retrieval')
L  = load('06_sklearn_lp')
E1 = load('02_explore')
KV = load('09_knn_variants')

for df, cols in [(A, ['sweep', 'it', 'K', 'lam', 'reg', 'k', 'alpha', 'fit_on', 'graph_from', 'lp_alpha', 'homophily']),
                 (R, ['subset', 'k', 'K', 'lam', 'it', 'm', 'reg', 'ppr_alpha', 'homophily'])]:
    for c in cols:
        if c not in df:
            df[c] = np.nan

DATASETS = [
    ('pets',          'pets',          37,  'Oxford-IIIT Pets: cat and dog breeds'),
    ('cars',          'cars',          196, 'Stanford Cars: make, model and year'),
    ('resisc45',      'resisc45',      45,  'NWPU-RESISC45: satellite scene types'),
    ('flowers',       'flowers',       102, 'Oxford Flowers: flower species'),
    ('food101',       'food101',       101, 'Food-101: dishes'),
    ('fashion_mnist', 'fashion_mnist', 10,  'Fashion-MNIST: 28×28 grayscale clothing items'),
]
DATASETS = [d for d in DATASETS if d[0] in set(A.dataset) | set(R.dataset)]
DS       = [d[0] for d in DATASETS]
ROWS     = [(d[0], d[1]) for d in DATASETS]
NPCS     = [1, 5]

S1, S2, S3, S4 = 'var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)'
MUTED, INK2    = 'var(--muted)', 'var(--ink-2)'

# --
# Classification: mean accuracy per dataset, over seeds

def acc(mask, npc):
    return A[mask & (A.npc == npc)].groupby('dataset').acc.mean().reindex(DS)

def hom(mask):
    return A[mask].groupby('dataset').homophily.mean().reindex(DS)

def m_ppr(pred='svc', k=10, alpha=0.85):
    return (A.method == 'ppr') & (A.k == k) & (A.alpha == alpha) & (A.pred == pred)

def m_km(pred='svc', it=1, sweep='default', **kw):
    m = (A.method == 'kmeans_agop') & (A.sweep == sweep) & (A['it'] == it) & (A.pred == pred)
    for c, v in kw.items():
        m &= A[c] == v
    return m

def m_simple(method, pred='svc', it=None):
    m = (A.method == method) & (A.pred == pred)
    if it is not None:
        m &= A['it'] == it
    return m

HEAD = {   # key -> (label, mask)
    'svc'          : ('SVM on raw vectors',                        m_simple('svc')),
    'ppr'          : ('Feature spreading (original method)',       m_ppr('svc')),
    'lp'           : ('Label propagation, same graph',             m_ppr('lp')),
    'km'           : ('Cluster-metric graph + SVM',                m_km('svc', 1)),
    'km_lp'        : ('Cluster-metric graph + label propagation',  m_km('lp', 1)),
    'km2'          : ('Cluster-metric graph + SVM, 2 rounds',      m_km('svc', 2)),
    'pl'           : ('Pseudo-label-metric graph + SVM',           m_simple('pl_agop', 'svc', 3)),
    'pl_lp'        : ('Pseudo-label-metric graph + label propagation', m_simple('pl_agop', 'lp', 3)),
    'plkm'         : ('Both targets + SVM',                        m_simple('pl+kmeans_agop', 'svc', 3)),
    'plkm_lp'      : ('Both targets + label propagation',          m_simple('pl+kmeans_agop', 'lp', 3)),
    'oracle_agop'  : ('Ceiling: metric from all true labels',      m_simple('oracle_agop')),
    'oracle_graph' : ('Ceiling: graph with no cross-class edges',  m_simple('oracle_graph')),
}
H = {k : {npc : acc(m, npc) for npc in NPCS} for k, (_, m) in HEAD.items()}

def hv(key, ds, npc):
    v = H[key][npc].get(ds, np.nan)
    return float(v) if v is not None else np.nan

def delta(key, ds, npc, base='ppr'):
    return hv(key, ds, npc) - hv(base, ds, npc)

def mean_delta(key, npc, base='ppr'):
    return float(np.nanmean([delta(key, d, npc, base) for d in DS]))

def n_wins(key, npc, base='ppr', tol=0.0):
    return int(sum(delta(key, d, npc, base) > tol for d in DS))

N_SEEDS = A.groupby('dataset').seed.nunique().reindex(DS)

# --
# Figures

def fig(html_, caption, wide=False, tbl=None, tbl_label='Show the numbers'):
    det = f'<details class="numbers"><summary>{tbl_label}</summary>{tbl}</details>' if tbl else ''
    return f'<figure class="{"wide" if wide else ""}">{html_}<figcaption>{caption}</figcaption>{det}</figure>'

def panels(*svgs, cls='panels'):
    return f'<div class="{cls}">' + ''.join(f'<div class="panel">{s}</div>' for s in svgs) + '</div>'

# Fig: headline -- change vs feature spreading
head_series = [
    dict(key='svc',         label='SVM on raw vectors (no graph)',          color=MUTED, kind='tick'),
    dict(key='km',          label='Cluster-metric graph',                   color=S1,    kind='dot'),
    dict(key='pl',          label='Pseudo-label-metric graph',              color=S2,    kind='dot'),
    dict(key='plkm',        label='Both targets',                           color=S3,    kind='dot'),
    dict(key='oracle_agop', label='Ceiling: metric from all true labels',   color=INK2,  kind='tick'),
]
def head_plot(npc):
    return dotplot(ROWS, head_series, lambda d, k: delta(k, d, npc), xlim=(-0.16, 0.20), signed=True, zero=0,
                   title=f'{npc} labeled example{"s" if npc > 1 else ""} per class')

def head_table(npc, keys):
    rows = [[HEAD[k][0]] + [hv(k, d, npc) for d in DS] + [float(np.nanmean([hv(k, d, npc) for d in DS]))] for k in keys]
    skip = [i for i, k in enumerate(keys) if k.startswith('oracle')]
    return table(['Method'] + DS + ['mean'], rows, best=best_cells(rows, range(1, len(DS) + 2), skip),
                 caption=f'Accuracy, {npc}-shot. Bold marks the best non-ceiling method per dataset.')

HEAD_KEYS = ['svc', 'ppr', 'lp', 'km', 'km_lp', 'km2', 'pl', 'pl_lp', 'plkm', 'plkm_lp', 'oracle_agop', 'oracle_graph']
FIG_HEAD  = fig(
    legend(head_series) + panels(head_plot(1), head_plot(5)),
    'Change in accuracy relative to the original feature-spreading method (the zero line), per dataset. '
    'Dots to the right of zero are improvements. The ceiling markers use every true label and are not real methods.',
    wide=True, tbl=head_table(1, HEAD_KEYS) + head_table(5, HEAD_KEYS), tbl_label='Show accuracy for every method',
)

# Fig: K sweep
def k_get(npc, d):
    def get(pred, K):
        if K == 100:
            return acc(m_km(pred, 1), npc).get(d, np.nan)
        return acc(m_km(pred, 1, 'K', K=K), npc).get(d, np.nan)
    return get

K_GRID   = [10, 20, 50, 100, 200, 400]
k_series = [dict(key='svc', label='Cluster-metric graph + SVM', color=S1, kind='line'),
            dict(key='lp',  label='Cluster-metric graph + label propagation', color=S2, kind='line')]
def k_fig(npc):
    ps = [linepanel(f'{d} ({c} classes)', K_GRID, k_series, k_get(npc, d), ref=hv('ppr', d, npc), ref_label='feature spreading',
                    logx=True, xlabel='clusters K', xticks=[10, 50, 200]) for d, _, c, _ in DATASETS]
    return panels(*ps, cls='panels panels-3')

def k_table(npc):
    rows = []
    for pred, name in [('svc', 'SVM'), ('lp', 'label prop.')]:
        for K in K_GRID:
            rows.append([f'K = {K}, {name}'] + [k_get(npc, d)(pred, K) for d in DS])
    rows.append(['feature spreading'] + [hv('ppr', d, npc) for d in DS])
    return table(['Clusters, predictor'] + DS, rows, caption=f'Accuracy by number of clusters, {npc}-shot, one round.')

FIG_K = fig(
    legend(k_series + [dict(key='ref', label='Feature spreading baseline', color=MUTED, kind='line')]) + k_fig(1),
    '1-shot accuracy as the number of k-means clusters changes, one panel per dataset (each with its own vertical scale). '
    'The gray line is the original method with its SVM. Everything else is at its default. For comparison with the orange line, '
    'label propagation on the plain graph scores ' + ', '.join(f'{f3(hv("lp", d, 1))} on {d}' for d in DS) + '.',
    wide=True, tbl=k_table(1) + k_table(5), tbl_label='Show the numbers (1-shot and 5-shot)',
)

# Fig: ablations / controls -- mean change vs feature spreading
ABL = [
    ('Default: k-means on smoothed features, K=100',      m_km('svc', 1)),
    ('… second round',                                    m_km('svc', 2)),
    ('… fourth round',                                    m_km('svc', 4)),
    ('Cluster and fit on raw vectors instead',            m_km('svc', 1, 'fit_on')),
    ('Build the new graph from smoothed features',        m_km('svc', 1, 'graph_from')),
    ('Pseudo-label targets (3 rounds)',                   m_simple('pl_agop', 'svc', 3)),
    ('Pseudo-labels + clusters (3 rounds)',               m_simple('pl+kmeans_agop', 'svc', 3)),
    ('Control: re-graph on smoothed features, no metric', m_simple('regraph_P', 'svc', 1)),
    ('Control: project onto top-100 principal directions', m_simple('pca_proj')),
    ('Control: whiten top-100 principal directions',      m_simple('pca_white')),
    ('Control: random targets',                           m_simple('random_agop')),
]
def abl_rows(npc):
    base = H['ppr'][npc]
    out  = []
    for label, m in ABL:
        d = acc(m, npc) - base
        out.append(dict(label=label, value=float(np.nanmean(d.values)), points=[(k, float(v)) for k, v in d.items()]))
    return out

def abl_table(npc):
    rows = [[r['label']] + [v for _, v in r['points']] + [r['value']] for r in abl_rows(npc)]
    rows = [[r[0]] + [sgn(v) for v in r[1:]] for r in rows]
    return table(['Variant'] + DS + ['mean'], rows, caption=f'Change in accuracy vs feature spreading, {npc}-shot.')

_pv      = [v for r in abl_rows(1) for _, v in r['points'] if not np.isnan(v)]
ABL_XLIM = (np.floor(min(_pv) * 20) / 20, np.ceil(max(_pv) * 20) / 20)
FIG_ABL = fig(
    '<div class="legend"><span class="legend-item"><span class="key key-bar" style="background:var(--pos)"></span>mean gain over datasets</span>'
    '<span class="legend-item"><span class="key key-bar" style="background:var(--neg)"></span>mean loss</span>'
    '<span class="legend-item"><span class="key key-pt"></span>one dataset</span></div>'
    + panels(divbars(abl_rows(1), ABL_XLIM, '1-shot ablations'), cls='panels panels-1'),
    '1-shot change in accuracy relative to feature spreading for each variant and control (SVM on the labeled points as the final classifier). '
    'Bars are the mean over datasets; small dots are individual datasets.',
    wide=True, tbl=abl_table(1) + abl_table(5), tbl_label='Show per-dataset numbers (1-shot and 5-shot)',
)

# Tables: one-factor sweeps
def sweep_table(col, values, sweep, default, fmt=lambda v: f'{v:g}', it=1):
    out = ''
    for npc in NPCS:
        rows = []
        for v in values:
            m = m_km('svc', it) if v == default else m_km('svc', it, sweep, **{col : v})
            a = acc(m, npc)
            rows.append([fmt(v) + (' (default)' if v == default else '')] + [float(a.get(d, np.nan)) for d in DS] + [float(np.nanmean(a.values))])
        out += table([col] + DS + ['mean'], rows, best=best_cells(rows, range(1, len(DS) + 2)), caption=f'{npc}-shot accuracy, cluster-metric graph + SVM, one round.')
    return out

def rounds_table():
    out = ''
    for npc in NPCS:
        rows = [['0 (feature spreading)'] + [hv('ppr', d, npc) for d in DS] + [float(np.nanmean([hv('ppr', d, npc) for d in DS]))]]
        for it in [1, 2, 3, 4]:
            a = acc(m_km('svc', it), npc)
            rows.append([str(it)] + [float(a.get(d, np.nan)) for d in DS] + [float(np.nanmean(a.values))])
        out += table(['Rounds'] + DS + ['mean'], rows, best=best_cells(rows, range(1, len(DS) + 2)), caption=f'{npc}-shot accuracy by number of rounds (cluster targets).')
    return out

def kalpha_table():
    out = ''
    for npc in NPCS:
        rows = []
        for k, alpha in [(5, 0.85), (10, 0.85), (20, 0.85), (10, 0.5), (10, 0.95)]:
            b  = acc(m_ppr('svc', k, alpha), npc)
            m  = m_km('svc', 1) if (k, alpha) == (10, 0.85) else m_km('svc', 1, 'k_alpha', k=k, alpha=alpha)
            a  = acc(m, npc)
            nm = f'k={k}, α={alpha:g}' + (' (default)' if (k, alpha) == (10, 0.85) else '')
            rows.append([nm + ', feature spreading'] + [float(b.get(d, np.nan)) for d in DS] + [float(np.nanmean(b.values))])
            rows.append([nm + ', cluster-metric graph'] + [float(a.get(d, np.nan)) for d in DS] + [float(np.nanmean(a.values))])
        out += table(['Graph settings, method'] + DS + ['mean'], rows, best=best_cells(rows, range(1, len(DS) + 2)), caption=f'{npc}-shot accuracy.')
    return out

def hom_table():
    rows = [
        ['kNN graph on raw vectors']                  + list(hom(m_ppr('svc')).values),
        ['Re-graph on smoothed features (control)']   + list(hom(m_simple('regraph_P', 'svc', 1)).values),
        ['Cluster-metric graph, 1 round']             + list(hom(m_km('svc', 1)).values),
        ['Cluster-metric graph, 2 rounds']            + list(hom(m_km('svc', 2)).values),
        ['Pseudo-label-metric graph, 1-shot']         + list(hom(m_simple('pl_agop', 'svc', 3) & (A.npc == 1)).values),
        ['Pseudo-label-metric graph, 5-shot']         + list(hom(m_simple('pl_agop', 'svc', 3) & (A.npc == 5)).values),
        ['Random-target metric (control)']            + list(hom(m_simple('random_agop')).values),
        ['Ceiling: metric from all true labels']      + list(hom(m_simple('oracle_agop')).values),
    ]
    rows = [[r[0]] + [float(v) for v in r[1:]] for r in rows]
    return table(['Graph'] + DS, rows, caption='Edge homophily: the fraction of graph edges that join two points of the same class.')

def kernel_table():
    ds_k = [d for d in DS if ((A.method == 'pl_kernel_agop') & (A.dataset == d)).any()]
    out  = ''
    for npc in NPCS:
        rows = []
        for label, m in [('Feature spreading', m_ppr('svc')), ('Linear AGOP, pseudo-labels, round 2', m_simple('pl_agop', 'svc', 2)),
                         ('Kernel AGOP, confident pseudo-labels, round 2', m_simple('pl_kernel_agop', 'svc', 2))]:
            a = acc(m, npc)
            rows.append([label] + [float(a.get(d, np.nan)) for d in ds_k])
        out += table(['Method'] + ds_k, rows, best=best_cells(rows, range(1, len(ds_k) + 1)), caption=f'{npc}-shot accuracy. Kernel AGOP was only run where the class count is 50 or fewer.')
    return out

# Fig: sklearn label propagation
LP_SERIES = [
    dict(key='sklearn LabelPropagation (directed kNN)',  label='sklearn LabelPropagation, as shipped',       color=MUTED, kind='tick'),
    dict(key='sklearn LabelSpreading (directed kNN)',    label='sklearn LabelSpreading, as shipped',         color=S2,    kind='dot'),
    dict(key='sklearn LabelSpreading (symmetrized kNN)', label='sklearn LabelSpreading, symmetrized graph',  color=S3,    kind='dot'),
    dict(key='PPR on one-hot labels (repo graph)',       label='PPR on one-hot labels, this repo’s graph',   color=S1,    kind='ring'),
]
def lp_acc(method, npc):
    return L[(L.method == method) & (L.npc == npc)].groupby('dataset').acc.mean().reindex(DS)

def lp_plot(npc):
    return dotplot(ROWS, LP_SERIES, lambda d, k: lp_acc(k, npc).get(d, np.nan), xlim=(0.0, 1.0),
                   title=f'{npc} labeled example{"s" if npc > 1 else ""} per class')

def lp_table(npc):
    ms   = ['sklearn LabelPropagation (directed kNN)', 'sklearn LabelPropagation (symmetrized kNN)',
            'sklearn LabelSpreading (directed kNN)', 'sklearn LabelSpreading (symmetrized kNN)', 'PPR on one-hot labels (repo graph)']
    rows = [[m] + [float(lp_acc(m, npc).get(d, np.nan)) for d in DS] for m in ms]
    return table(['Method'] + DS, rows, best=best_cells(rows, range(1, len(DS) + 1)), caption=f'{npc}-shot accuracy.')

def lp_diag(col):
    return L[L.method == 'sklearn LabelSpreading (directed kNN)'].groupby('dataset')[col].mean().reindex(DS)

FIG_LP = fig(
    legend(LP_SERIES) + panels(lp_plot(1), lp_plot(5)),
    'Accuracy of four label-propagation variants on the same vectors. The only difference between the orange and green dots is whether the '
    'neighbor graph is symmetrized. Where the green dot sits inside the blue ring, the two methods score the same.', wide=True, tbl=lp_table(1) + lp_table(5),
)

# --
# Retrieval

def rmap(graph, scorer, npc, val='map', **kw):
    m = (R.graph == graph) & (R.scorer == scorer) & R.subset.isna() & (R.npc == npc)
    for c, v in kw.items():
        m &= R[c].isna() if v is None else (R[c] == v)
    return R[m].groupby('dataset')[val].mean().reindex(DS)

KM = dict(K=100, lam=0.5, it=1)
RET = {   # key -> (label, fn(npc, val))
    'cos'        : ('Cosine similarity, raw vectors',                 lambda n, v: rmap('none', 'cos', n, v)),
    'svc'        : ('One-vs-rest SVM, raw vectors',                   lambda n, v: rmap('none', 'svc1vr', n, v)),
    'elda'       : ('One-vs-rest ridge (penalty 10), raw vectors',    lambda n, v: rmap('none', 'elda', n, v, reg=10.0)),
    'aqe'        : ('Query expansion (top 10), raw vectors',          lambda n, v: rmap('none', 'aqe', n, v, m=10)),
    'knn_cos'    : ('Cosine on spread features',                      lambda n, v: rmap('knn', 'cos', n, v, k=10)),
    'knn_svc'    : ('One-vs-rest SVM on spread features',             lambda n, v: rmap('knn', 'svc1vr', n, v, k=10)),
    'knn_elda'   : ('One-vs-rest ridge on spread features',           lambda n, v: rmap('knn', 'elda', n, v, k=10)),
    'knn_ppr'    : ('PageRank from the exemplar, kNN graph',          lambda n, v: rmap('knn', 'ppr', n, v, k=10, ppr_alpha=0.95)),
    're_ppr'     : ('PageRank, graph rebuilt on smoothed features',   lambda n, v: rmap('regraph_P', 'ppr', n, v, ppr_alpha=0.95)),
    'km_cos'     : ('Cosine on features spread over cluster-metric graph', lambda n, v: rmap('kmeans_agop', 'cos', n, v, **KM)),
    'km_svc'     : ('One-vs-rest SVM, cluster-metric graph',          lambda n, v: rmap('kmeans_agop', 'svc1vr', n, v, **KM)),
    'km_elda'    : ('One-vs-rest ridge, cluster-metric graph',        lambda n, v: rmap('kmeans_agop', 'elda', n, v, **KM)),
    'km_ppr'     : ('PageRank, cluster-metric graph',                 lambda n, v: rmap('kmeans_agop', 'ppr', n, v, ppr_alpha=0.95, **KM)),
    'km2_ppr'    : ('PageRank, cluster-metric graph, 2 rounds',       lambda n, v: rmap('kmeans_agop', 'ppr', n, v, ppr_alpha=0.95, K=100, lam=0.5, it=2)),
    'metric_cos' : ('Cosine under the cluster metric, no graph',      lambda n, v: rmap('none', 'cos_metric', n, v)),
    'or_ppr'     : ('Ceiling: PageRank, metric from all true labels', lambda n, v: rmap('oracle_agop', 'ppr', n, v, ppr_alpha=0.95)),
}
RV = {k : {(n, v) : f(n, v) for n in NPCS for v in ['map', 'p10']} for k, (_, f) in RET.items()}

def rv(key, ds, npc, val='map'):
    return float(RV[key][(npc, val)].get(ds, np.nan))

def rmean(key, npc, val='map'):
    return float(np.nanmean([rv(key, d, npc, val) for d in DS]))

ret_series = [
    dict(key='cos',     label='Cosine similarity, raw vectors',           color=MUTED, kind='tick'),
    dict(key='knn_cos', label='Cosine on spread features',                color=S2,    kind='dot'),
    dict(key='knn_ppr', label='PageRank from the exemplar, kNN graph',    color=S3,    kind='dot'),
    dict(key='km_ppr',  label='PageRank, cluster-metric graph',           color=S1,    kind='dot'),
    dict(key='or_ppr',  label='Ceiling: metric from all true labels',     color=INK2,  kind='tick'),
]
def ret_plot(npc):
    return dotplot(ROWS, ret_series, lambda d, k: rv(k, d, npc), xlim=(0.0, 1.0),
                   title=f'{npc} exemplar{"s" if npc > 1 else ""} of the target class')

def ret_table(npc, val):
    keys = list(RET)
    rows = [[RET[k][0]] + [rv(k, d, npc, val) for d in DS] + [rmean(k, npc, val)] for k in keys]
    skip = [i for i, k in enumerate(keys) if k.startswith('or_')]
    name = 'Mean average precision' if val == 'map' else 'Precision in the top 10'
    return table(['Scorer'] + DS + ['mean'], rows, best=best_cells(rows, range(1, len(DS) + 2), skip), caption=f'{name}, {npc} exemplar{"s" if npc > 1 else ""}.')

FIG_RET = fig(
    legend(ret_series) + panels(ret_plot(1), ret_plot(5)),
    'Mean average precision when ranking all points for one target class. Higher is better; 1.0 means every member of the class is ranked above every non-member.',
    wide=True, tbl=ret_table(1, 'map') + ret_table(5, 'map') + ret_table(1, 'p10') + ret_table(5, 'p10'), tbl_label='Show every scorer (average precision and precision@10)',
)

def ret_k_table():
    out = ''
    for npc in NPCS:
        rows = []
        for K in [20, 50, 100, 200, 400]:
            a = rmap('kmeans_agop', 'ppr', npc, ppr_alpha=0.95, K=K, lam=0.5, it=1)
            rows.append([f'K = {K}'] + [float(a.get(d, np.nan)) for d in DS] + [float(np.nanmean(a.values))])
        b = RV['knn_ppr'][(npc, 'map')]
        rows.append(['kNN graph (no metric)'] + [float(b.get(d, np.nan)) for d in DS] + [float(np.nanmean(b.values))])
        out += table(['Clusters'] + DS + ['mean'], rows, best=best_cells(rows, range(1, len(DS) + 2)), caption=f'Mean average precision, PageRank scorer, {npc} exemplar{"s" if npc > 1 else ""}.')
    return out

def ret_alpha_table():
    out = ''
    for npc in NPCS:
        rows = []
        for g, kw, name in [('knn', dict(k=10), 'kNN graph'), ('kmeans_agop', KM, 'cluster-metric graph')]:
            for a_ in [0.85, 0.95, 0.99]:
                a = rmap(g, 'ppr', npc, ppr_alpha=a_, **kw)
                rows.append([f'{name}, α = {a_:g}'] + [float(a.get(d, np.nan)) for d in DS] + [float(np.nanmean(a.values))])
        out += table(['Graph, restart'] + DS + ['mean'], rows, best=best_cells(rows, range(1, len(DS) + 2)), caption=f'Mean average precision, PageRank scorer, {npc} exemplar{"s" if npc > 1 else ""}.')
    return out

def ret_query_table():
    Q = R[R.subset == 'q20']
    def q(graph, scorer, npc, **kw):
        m = (Q.graph == graph) & (Q.scorer == scorer) & (Q.npc == npc)
        for c, v in kw.items():
            m &= Q[c] == v
        return Q[m].groupby('dataset')['map'].mean().reindex(DS)
    out = ''
    for npc in NPCS:
        rows = []
        for name, a in [('PageRank, kNN graph', q('knn', 'ppr', npc)),
                        ('PageRank, per-query metric graph, λ = 0.02', q('query_agop', 'ppr', npc, lam=0.02)),
                        ('PageRank, per-query metric graph, λ = 0.1',  q('query_agop', 'ppr', npc, lam=0.1))]:
            rows.append([name] + [float(a.get(d, np.nan)) for d in DS] + [float(np.nanmean(a.values))])
        out += table(['Scorer'] + DS + ['mean'], rows, best=best_cells(rows, range(1, len(DS) + 2)),
                     caption=f'Mean average precision on a 20-query subset per seed, {npc} exemplar{"s" if npc > 1 else ""}.')
    return out

# --
# Toy figure: what spreading does (real computation, 2-D so it can be drawn)

def toy_fig():
    rng  = np.random.default_rng(3)
    cen  = np.array([[0, 0], [3.2, 0.4], [1.5, 2.9]])
    lab  = np.repeat([0, 1, 2], 45)
    X    = cen[lab] + rng.normal(0, 0.85, (len(lab), 2))
    n    = len(X)
    d2   = ((X[:,None] - X[None]) ** 2).sum(-1)
    k    = 6
    nn   = np.argsort(d2, 1)[:,1:k + 1]
    adj  = np.zeros((n, n))
    adj[np.repeat(np.arange(n), k), nn.ravel()] = 1
    adj  = np.maximum(adj, adj.T)
    Ah   = adj + np.eye(n)
    dinv = 1 / np.sqrt(Ah.sum(1))
    Ah   = Ah * dinv[:,None] * dinv[None,:]
    P    = 0.15 * np.linalg.solve(np.eye(n) - 0.85 * Ah, X)
    P    = P / (0.15 * np.linalg.solve(np.eye(n) - 0.85 * Ah, np.ones(n)))[:,None]   # undo degree scaling so it stays drawable
    edges  = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
    marked = [int(np.argmin(((X - c) ** 2).sum(1) + 1e9 * (lab != i))) for i, c in enumerate(cen + [[-0.9, 0.5], [0.8, -0.7], [0.6, 0.9]])]
    lim    = (X[:,0].min() - 0.3, X[:,0].max() + 0.3, X[:,1].min() - 0.3, X[:,1].max() + 0.3)
    same   = np.mean([lab[i] == lab[j] for i, j in edges])
    return panels(toy(X, lab, edges, marked, 'Raw vectors and their neighbor graph', lim),
                  toy(P, lab, [], marked, 'After spreading (α = 0.85)', lim), cls='panels panels-toy'), same

TOY, TOY_HOM = toy_fig()
FIG_TOY = fig(
    '<div class="legend"><span class="legend-item"><span class="key key-dot" style="background:var(--s1)"></span>class A</span>'
    '<span class="legend-item"><span class="key key-dot" style="background:var(--s2)"></span>class B</span>'
    '<span class="legend-item"><span class="key key-dot" style="background:var(--s3)"></span>class C</span>'
    '<span class="legend-item"><span class="key key-ring"></span>the one labeled point per class</span>'
    '<span class="legend-item"><span class="key key-line" style="background:var(--neg)"></span>edge joining two different classes</span></div>' + TOY,
    f'A two-dimensional toy example with three overlapping classes. Left: each point is joined to its 6 nearest neighbors; {TOY_HOM:.0%} of the edges '
    'join two points of the same class. Right: the same points after each one is replaced by a PageRank-weighted average of its graph neighborhood. '
    'The classes pull apart, so a classifier trained on the three ringed points now has a much easier job. Colors show the true classes, which the method never sees.',
)

# ==
# Text

# --
# Stage-1 negative results (pets only, 5 seeds) and the very first AGOP loop (pets, 1 seed)

E0 = load('01_agop_sweep')

def e1(mask, npc):
    return float(E1[mask & (E1.npc == npc)].groupby('seed').acc.mean().mean())

def e1_pl(kind, where, fit, npc, q=1.0, lam=0.5, it=1):
    return e1((E1.method == 'pl_metric') & (E1.kind == kind) & (E1['where'] == where) & (E1.fit == fit) & (E1.q == q) & (E1.lam == lam) & (E1['it'] == it), npc)

def e1_unsup_best(npc):
    u = E1[(E1.method == 'ppr_unsup') & (E1.npc == npc)]
    return float(u.groupby(['g_feat', 's_feat', 'k', 'weighted', 'alpha']).acc.mean().max())

E1_ROWS = [
    ('SVM on raw vectors',                                             lambda n: e1(E1.method == 'svc', n)),
    ('Feature spreading (original method)',                            lambda n: e1(E1.method == 'ppr', n)),
    ('Best of 72 unsupervised graph settings (centering, k, α, edge weights)', e1_unsup_best),
    ('Self-training on pseudo-labels, graph unchanged',                lambda n: e1((E1.method == 'selftrain') & (E1.q == 1.0) & (E1['it'] == 1), n)),
    ('Delete edges between points with different predicted labels',    lambda n: e1((E1.method == 'pl_prune') & (E1.q == 0.5) & (E1['it'] == 1) & (E1.fit == 'lab'), n)),
    ('Class-mean metric from pseudo-labels, applied to the graph',     lambda n: e1_pl('ncm', 'graph', 'lab', n)),
    ('Ridge-AGOP metric from pseudo-labels, applied to the features only', lambda n: e1_pl('linear', 'feat', 'lab', n)),
    ('Ridge-AGOP metric from pseudo-labels, applied to graph and features', lambda n: e1_pl('linear', 'both', 'lab', n)),
    ('Ridge-AGOP metric from pseudo-labels, applied to the graph',     lambda n: e1_pl('linear', 'graph', 'lab', n)),
    ('… same graph, final classifier also trained on pseudo-labels',   lambda n: e1_pl('linear', 'graph', 'pl', n)),
]
_rows    = [[name, f(1), f(5)] for name, f in E1_ROWS]
E1_TABLE = table(['Variant (pets, 5 seeds)', '1-shot', '5-shot'], _rows, best=best_cells(_rows, [1, 2]),
                 caption='Accuracy on pets. Final classifier is an SVM on the labeled points unless stated.')

def e0(method, lam=None, reg=None, it=None, col='acc'):
    m = E0.method == method
    if lam is not None: m &= E0.lam == lam
    if reg is not None: m &= E0.reg == reg
    if it  is not None: m &= E0['it'] == it
    return float(E0[m][col].mean())

# --
# Numbers quoted in the prose

def p(v):  return f'{v:.3f}'
def pts(v): return f'{100 * v:+.1f}'
def rng_(key, npc, base='ppr'):
    d = [delta(key, x, npc, base) for x in DS]
    return f'{100 * np.nanmin(d):+.1f} to {100 * np.nanmax(d):+.1f}'

n_ds       = len(DS)
seeds_note = ', '.join(f'{d} {int(N_SEEDS[d])}' for d in DS)

BEST1 = {d : max(['km', 'km_lp', 'pl', 'pl_lp', 'plkm', 'plkm_lp'], key=lambda k: hv(k, d, 1)) for d in DS}

LP_NEVER  = lp_diag('frac_labeled_never_listed')
LP_SK     = {n : lp_acc('sklearn LabelSpreading (directed kNN)', n)    for n in NPCS}
LP_SYM    = {n : lp_acc('sklearn LabelSpreading (symmetrized kNN)', n) for n in NPCS}
LP_SKLP   = {n : lp_acc('sklearn LabelPropagation (directed kNN)', n)  for n in NPCS}
LP_PPR    = {n : lp_acc('PPR on one-hot labels (repo graph)', n)       for n in NPCS}
mean_     = lambda s: float(np.nanmean(s.values))

CSS = r"""
/* Layout: one reading column (44rem) inside a 62rem frame; figures and tables use the full frame. */
:root {
  --plane: #f9f9f7; --surface: #fcfcfb; --ink: #1a1a18; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --code-bg: #f0efec; --link: #1c5cab;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --pos: #2a78d6; --neg: #e34948;
  --font: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --plane: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --code-bg: #232321; --link: #6da7ec;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --pos: #3987e5; --neg: #e66767;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --plane: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --code-bg: #232321; --link: #6da7ec;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --pos: #3987e5; --neg: #e66767;
  color-scheme: dark;
}
* { box-sizing: border-box }
html { -webkit-text-size-adjust: 100% }
body { margin: 0; background: var(--plane); color: var(--ink); font: 16px/1.62 var(--font); }
.frame { max-width: 62rem; margin: 0 auto; padding-block: 2.5rem 5rem;
         padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.frame > :where(p, h1, h2, h3, ul, ol, pre, nav, .lede, .note) { max-width: 44rem; }
h1 { font-size: 2.1rem; line-height: 1.15; margin: 0 0 .6rem; letter-spacing: -0.02em; text-wrap: balance; }
h2 { font-size: 1.4rem; line-height: 1.25; margin: 3.2rem 0 .8rem; padding-top: 1.4rem; border-top: 1px solid var(--grid); letter-spacing: -0.01em; text-wrap: balance; max-width: none !important; }
h3 { font-size: 1.05rem; margin: 2rem 0 .4rem; text-wrap: balance; }
p { margin: 0 0 1rem; }
.lede { font-size: 1.12rem; color: var(--ink-2); margin-bottom: 1.6rem; }
.meta { color: var(--muted); font-size: .85rem; margin-bottom: 2rem; }
a { color: var(--link); }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--link); outline-offset: 2px; border-radius: 2px; }
ul, ol { padding-left: 1.3rem; margin: 0 0 1rem; }
li { margin-bottom: .45rem; }
code, pre { font-family: var(--mono); font-size: .86em; }
code { background: var(--code-bg); padding: .1em .32em; border-radius: 4px; }
pre { background: var(--code-bg); padding: .9rem 1rem; border-radius: 8px; overflow-x: auto; line-height: 1.5; max-width: 62rem !important; }
pre code { background: none; padding: 0; font-size: .82rem; }
.eq { font-family: var(--mono); font-size: .92rem; background: var(--code-bg); border-radius: 8px; padding: .7rem 1rem; margin: 0 0 1rem; overflow-x: auto; white-space: nowrap; max-width: 44rem; }
nav.toc { font-size: .9rem; margin: 0 0 2rem; }
nav.toc ol { columns: 2; column-gap: 2rem; padding-left: 1.2rem; margin: .4rem 0 0; }
nav.toc li { margin-bottom: .2rem; break-inside: avoid; }
@media (max-width: 34rem) { nav.toc ol { columns: 1 } h1 { font-size: 1.7rem } }
.findings { list-style: none; padding: 0; display: grid; gap: .7rem; max-width: 44rem; }
.findings li { margin: 0; padding-left: 1rem; border-left: 3px solid var(--s1); }
.findings li.warn { border-left-color: var(--neg); }
.note { color: var(--ink-2); font-size: .92rem; border-left: 3px solid var(--axis); padding-left: 1rem; margin: 0 0 1rem; }
ol.steps { counter-reset: s; list-style: none; padding: 0; display: grid; gap: .55rem; }
ol.steps li { counter-increment: s; display: grid; grid-template-columns: 1.9rem 1fr; gap: .2rem .5rem; margin: 0; }
ol.steps li::before { content: counter(s); font: 600 .8rem/1.7rem var(--font); width: 1.7rem; height: 1.7rem; text-align: center; border-radius: 50%; background: var(--code-bg); color: var(--ink-2); font-variant-numeric: tabular-nums; }
ol.steps li > div { min-width: 0; }

figure { margin: 1.6rem 0 2rem; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.1rem 1.2rem 1rem; }
figure:not(.wide) { max-width: 44rem; }
figcaption { color: var(--ink-2); font-size: .88rem; margin-top: .8rem; max-width: 46rem; }
.legend { display: flex; flex-wrap: wrap; gap: .3rem 1.1rem; font-size: .82rem; color: var(--ink-2); margin-bottom: .8rem; }
.legend-item { display: inline-flex; align-items: center; gap: .4rem; }
.key { display: inline-block; flex: none; }
.key-dot { width: 10px; height: 10px; border-radius: 50%; }
.key-tick { width: 3px; height: 14px; border-radius: 1.5px; margin: 0 3px; }
.key-line { width: 16px; height: 2px; border-radius: 1px; }
.key-bar { width: 14px; height: 9px; border-radius: 2px; }
.key-pt { width: 6px; height: 6px; border-radius: 50%; background: var(--ink-2); }
.key-ring { width: 12px; height: 12px; border-radius: 50%; border: 1.5px solid var(--ink); }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 21rem), 1fr)); gap: 1rem 1.6rem; }
.panels-3 { grid-template-columns: repeat(auto-fit, minmax(min(100%, 14rem), 1fr)); }
.panels-1 { grid-template-columns: minmax(0, 46rem); }
.panel { min-width: 0; overflow-x: auto; }
.panels-toy { grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr)); }
.panels:not(.panels-3):not(.panels-toy) .panel svg { min-width: 24rem; }
.chart { width: 100%; height: auto; display: block; font-family: var(--font); }
.chart text { fill: var(--ink-2); font-size: 11px; }
.chart .ptitle { fill: var(--ink); font-size: 12px; font-weight: 600; }
.chart .tick { fill: var(--muted); font-size: 10.5px; font-variant-numeric: tabular-nums; }
.chart .rowlab { fill: var(--ink); font-size: 12px; }
.chart .val { fill: var(--ink); font-size: 11px; font-variant-numeric: tabular-nums; }
.chart .grid { stroke: var(--grid); stroke-width: 1; }
.chart .axis { stroke: var(--axis); stroke-width: 1.5; }
.chart .span { stroke: var(--axis); stroke-width: 1; }
.chart .ref { stroke: var(--muted); stroke-width: 1.5; }
.chart .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .dot { stroke: var(--surface); stroke-width: 2; }
.chart .dot-s { stroke: var(--surface); stroke-width: 1; }
.chart .pt { fill: var(--ink-2); stroke: var(--surface); stroke-width: 1; }
.chart .hit { fill: transparent; }
.chart .edge { stroke: var(--axis); stroke-width: .7; }
.chart .edge-x { stroke: var(--neg); stroke-width: 1; }
.chart .lab-ring { fill: none; stroke: var(--ink); stroke-width: 1.5; }
.chart .ring { fill: none; stroke-width: 2.5; }
.chart .mark { outline: none; cursor: default; }
.chart .mark:hover .dot, .chart .mark:focus-visible .dot, .chart .mark:hover .pt, .chart .mark:focus-visible .pt { stroke: var(--ink); }
.chart .mark:hover path, .chart .mark:focus-visible path, .chart .mark:hover rect:not(.hit), .chart .mark:focus-visible rect:not(.hit) { filter: brightness(1.15); }
#tip { position: fixed; z-index: 20; pointer-events: none; background: var(--ink); color: var(--plane); font-size: .78rem; line-height: 1.35; padding: .35rem .55rem; border-radius: 6px; max-width: 18rem; }

details.numbers { margin-top: .8rem; font-size: .88rem; }
details.numbers summary { cursor: pointer; color: var(--link); width: fit-content; }
.tablewrap { overflow-x: auto; margin: .8rem 0 1.2rem; }
table { border-collapse: collapse; font-size: .82rem; font-variant-numeric: tabular-nums; min-width: 100%; }
caption { text-align: left; color: var(--ink-2); padding-bottom: .4rem; font-size: .82rem; }
th, td { padding: .32rem .7rem .32rem 0; border-bottom: 1px solid var(--grid); text-align: left; white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; border-bottom-color: var(--axis); }
td.num, th.num { text-align: right; padding-right: 0; padding-left: .9rem; }
td b { font-weight: 700; }
.frame > .tablewrap { max-width: 62rem; }
dl.gloss { display: grid; grid-template-columns: minmax(8rem, 13rem) 1fr; gap: .5rem 1.2rem; max-width: 52rem; margin: 0; }
dl.gloss dt { font-weight: 600; } dl.gloss dd { margin: 0; color: var(--ink-2); }
@media (max-width: 34rem) { dl.gloss { grid-template-columns: 1fr; gap: .1rem; } dl.gloss dd { margin-bottom: .6rem; } }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth } }
"""

JS = r"""
(function () {
  var tip = document.getElementById('tip');
  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, x + 12)) + 'px';
    tip.style.top  = (y - h - 10 < 8 ? y + 16 : y - h - 10) + 'px';
  }
  function show(el, x, y) { tip.textContent = el.getAttribute('data-tip'); tip.hidden = false; place(x, y); }
  document.addEventListener('pointermove', function (e) {
    var m = e.target.closest && e.target.closest('[data-tip]');
    if (m) show(m, e.clientX, e.clientY); else tip.hidden = true;
  });
  document.addEventListener('focusin', function (e) {
    var m = e.target.closest && e.target.closest('[data-tip]');
    if (!m) return;
    var r = m.getBoundingClientRect();
    show(m, r.left + r.width / 2, r.top);
  });
  document.addEventListener('focusout', function () { tip.hidden = true; });
  document.addEventListener('scroll', function () { tip.hidden = true; }, { passive: true });
})();
"""

BODY = f"""
<div class="frame">

<h1>Feature spreading with a learned graph</h1>
<p class="lede">How to classify, or search, a pile of vectors when only one to five examples per class are labeled, and what a learned distance metric adds to the graph-smoothing method this repository started with.</p>
<p class="meta">Six datasets of 768-dimensional vectors · up to 5,000 points each · 1 and 5 labeled examples per class · seeds per dataset: {seeds_note}</p>

<h2 id="summary">Summary</h2>
SUMMARY_PLACEHOLDER

<nav class="toc" aria-label="Contents">
<strong>Contents</strong>
<ol>
<li><a href="#problem">The problem</a></li>
<li><a href="#background">Background: graphs, spreading, AGOP</a></li>
<li><a href="#first">The first attempt, and why it stalled</a></li>
<li><a href="#method">The method that works</a></li>
<li><a href="#results">Classification results</a></li>
<li><a href="#ablations">Tuning and ablations</a></li>
<li><a href="#lp">Label propagation, and why sklearn’s fails here</a></li>
<li><a href="#retrieval">One-class retrieval</a></li>
<li><a href="#recipe">Recommended recipe</a></li>
<li><a href="#caveats">Caveats</a></li>
<li><a href="#repro">Reproducing this</a></li>
<li><a href="#glossary">Glossary</a></li>
</ol>
</nav>

<h2 id="problem">1. The problem</h2>
<p>You have a few thousand vectors. Each one belongs to a class, but you only know the class for a handful of them: one per class (“1-shot”) or five per class (“5-shot”). The job is to label the rest.</p>
<p>The setting is <em>transductive</em>: the unlabeled vectors are all available up front, and they are the only points that need labels. That matters, because the unlabeled points carry information. They show where the data is dense, which points sit near each other, and where the natural groups are. A method that ignores them throws that away.</p>
<p>The vectors here are image embeddings from a CLIP model, but nothing below uses that. No images, no text, no class names. They are stand-ins for “arbitrary vectors with class structure”, and every method works on any such data.</p>
<div class="tablewrap">{table(['Dataset', 'Classes', 'What the classes are'], [[d[1], str(d[2]), d[3]] for d in DATASETS], num_from=1)}</div>
<p>The plain baseline is a linear SVM trained on the labeled vectors alone. With one example per class it has very little to go on.</p>

<h2 id="background">2. Background: graphs, spreading, AGOP</h2>
<p>The method is built from three ideas. Each is simple on its own.</p>

<h3>A neighbor graph</h3>
<p>Connect every point to the handful of points most similar to it, using cosine similarity. Make the connection mutual: if either point lists the other, they are linked. The result is a <em>k-nearest-neighbor graph</em>. It encodes the assumption that nearby points tend to share a class.</p>
<p class="note">How many neighbors: the code’s setting is <code>k = 10</code>, but because of a strict “greater than” comparison that counts the point itself, each point actually lists <strong>8</strong> neighbors, and its self-loop is counted twice. Every result in this report was produced that way, and “k” below always means the code’s setting (so k = 5 is 3 neighbors and k = 20 is 18). Section 6 checks that this makes little difference.</p>
<p>How true that assumption is can be measured, if you cheat and look at all the labels: <em>edge homophily</em> is the fraction of edges that join two points of the same class. It is a diagnostic used throughout this report. No method here gets to see it.</p>

<h3>Spreading: personalized PageRank over the graph</h3>
<p>Given the graph, replace each point’s vector with a weighted average of the vectors around it. Close graph neighbors get a large weight, neighbors of neighbors a smaller one, and so on. The specific weights used here are personalized PageRank (PPR): imagine a random walker that starts at a point, and at each step either moves to a random neighbor (with probability α) or jumps back to the start (with probability 1 − α). The weight on any other point is how much time the walker spends there.</p>
<p>All of those weights for all points at once have a closed form. With <code>Â</code> the adjacency matrix (plus self-loops) scaled by node degree, and <code>X</code> the matrix of vectors:</p>
<div class="eq">P = (1 − α) · (I − α·Â)⁻¹ · X</div>
<p>α sets the reach. At α = 0 nothing moves. Near 1, every point in a connected cluster collapses to the same average. The repository uses α = 0.85.</p>
{FIG_TOY}
<p>What you push through the graph is a choice:</p>
<ul>
<li><strong>Feature spreading</strong> pushes the vectors themselves, then trains an ordinary classifier on the labeled rows of the smoothed matrix <code>P</code>. This is the method the repository was written to test.</li>
<li><strong>Label propagation</strong> pushes the labels. Put a 1 in the right class column for each labeled point, zeros elsewhere, spread that matrix, and give each point the class with the largest value. Same graph, same formula, different payload.</li>
</ul>

<h3>AGOP: which directions does a predictor care about?</h3>
<p>Not every direction in a 768-dimensional space is useful for telling classes apart. A distance that weighs all of them equally lets irrelevant variation decide who is whose neighbor.</p>
<p>Take any trained predictor <code>f</code>. Its gradient at a point says which small changes to the input would change the prediction. The <em>average gradient outer product</em> (AGOP) averages <code>∇f · ∇fᵀ</code> over the data. The result is a d × d matrix <code>G</code> that is large along directions the predictor is sensitive to and near zero along directions it ignores.</p>
<p><em>Recursive feature machines</em> (RFM) use this as a feedback loop: fit a predictor, compute its AGOP, use the AGOP as the distance metric for the next fit, repeat. It is a way for a kernel method to learn which features matter, which is something neural networks do and plain kernel methods do not.</p>
<p>For a linear predictor <code>f(x) = W·x</code> the gradient is <code>W</code> everywhere, so the AGOP is just <code>WᵀW</code>. No sampling, no kernel. That is the version this report ends up using.</p>
<p>Using <code>G</code> as a metric means measuring similarity after mapping each vector through its square root, <code>x → G^½·x</code>, which stretches the directions the predictor uses and shrinks the rest. A damping weight λ blends it with the ordinary metric so it cannot collapse the space entirely:</p>
<div class="eq">M = (1 − λ)·I + λ·G / mean-eigenvalue(G)</div>

<h2 id="first">3. The first attempt, and why it stalled</h2>
<p>The starting idea was a direct RFM loop around feature spreading:</p>
<ol class="steps">
<li><div>Build the neighbor graph and spread the features.</div></li>
<li><div>Fit a predictor on the few labeled points.</div></li>
<li><div>Compute its AGOP and turn it into a metric.</div></li>
<li><div>Rebuild the neighbor graph under that metric. Go to 1.</div></li>
</ol>
<p>On pets at 1-shot (one seed) feature spreading scored {p(e0('ppr'))}. Four rounds of the loop with a kernel predictor landed between {p(min(e0('agop_kernel', l, r, 4) for l, r in [(0.25, 0.001), (0.25, 0.1), (0.5, 0.001)]))} and {p(max(e0('agop_kernel', l, r, 4) for l, r in [(0.25, 0.001), (0.25, 0.1), (0.5, 0.001)]))} depending on settings, and edge homophily moved from {p(e0('ppr', col='homophily'))} to about {p(e0('agop_kernel', 0.25, 0.001, 4, 'homophily'))}. At best that is three points on a single seed, from a graph that barely changed, and other settings gave nothing.</p>
<p>Two ceiling experiments showed the idea was sound and the execution was the problem. Both cheat by using every true label, so they are upper bounds and not methods:</p>
<ul>
<li>Delete every edge that joins two different classes, then run feature spreading as usual: {p(e0('oracle_graph'))}. So the graph is the bottleneck, with about {100 * (e0('oracle_graph') - e0('ppr')):.0f} points of headroom.</li>
<li>Compute the AGOP from a classifier trained on all labels, rebuild the graph with it, then run the normal 1-shot method: {p(e0('oracle_agop_linear', 1.0))}. So an AGOP metric <em>can</em> deliver most of that headroom.</li>
</ul>
<p>The loop stalls because of what feeds it. A predictor fit on one point per class only knows about those few points. For a linear model this is exact: its weights are a combination of its training vectors, so its AGOP spans at most as many directions as there are labeled points, and those directions describe the labeled points, not the classes. A kernel model has the same limitation in a softer form, since everything it knows about the labels still comes from those few points. Rebuilding the graph with that metric mostly connects points the classifier already thought were similar. Nothing new enters the loop.</p>
<p>The fix follows from that diagnosis: give the predictor a target for <em>every</em> point, not just the labeled ones.</p>

<h2 id="method">4. The method that works</h2>
<ol class="steps">
<li><div><strong>Graph and spread.</strong> Build the neighbor graph on the raw vectors <code>X</code> and spread them to get <code>P</code>, exactly as before.</div></li>
<li><div><strong>Make a target for every point.</strong> Either run k-means on <code>P</code> and use the cluster assignments (no labels involved), or train the few-shot classifier on <code>P</code> and use its predictions for all points as <em>pseudo-labels</em>. Or use both side by side.</div></li>
<li><div><strong>Fit a linear model to those targets.</strong> Ridge regression from <code>P</code> to the one-hot targets, which is a single linear solve. Call the weights <code>W</code>.</div></li>
<li><div><strong>Take its AGOP as a metric.</strong> <code>G = WᵀW</code>, damped with λ = 0.5.</div></li>
<li><div><strong>Rebuild the graph under the metric</strong> on the raw vectors, and spread the raw vectors over the new graph.</div></li>
<li><div><strong>Classify.</strong> Train the SVM on the labeled rows of the new <code>P</code>, or run label propagation on the new graph. Optionally repeat from step 2.</div></li>
</ol>
<p>The metric only decides who is whose neighbor. The vectors being averaged are still the original ones. And when the targets are k-means clusters, the graph is built without any labels at all; the labels are only used by the final classifier.</p>
<p>Why should cluster assignments work as targets? The clusters are found in the <em>smoothed</em> space, where spreading has already pulled natural groups together. A linear model that can predict those groups has to pick out the directions along which the groups differ. Those directions are a good guess at the ones that separate the real classes too, and the metric emphasizes exactly them.</p>

<h2 id="results">5. Classification results</h2>
RESULTS_PLACEHOLDER

<h2 id="ablations">6. Tuning and ablations</h2>
ABLATIONS_PLACEHOLDER

<h2 id="lp">7. Label propagation, and why sklearn’s fails here</h2>
LP_PLACEHOLDER

<h2 id="retrieval">8. One-class retrieval</h2>
RETRIEVAL_PLACEHOLDER

<h2 id="recipe">9. Recommended recipe</h2>
RECIPE_PLACEHOLDER

<h2 id="caveats">10. Caveats</h2>
CAVEATS_PLACEHOLDER

<h2 id="repro">11. Reproducing this</h2>
<p>Everything runs from the repository root. Features are downloaded and extracted once; the experiments only read them.</p>
<pre><code>pixi install
pixi run ./run.sh                                   # download + featurize all six datasets, run fs run / fs agop

# experiments (each reads data/, writes one CSV)
python experiments/04_ablate.py    pets 0 experiments/results/04_ablate/pets_0.csv        # classification: tuning + ablations
python experiments/05_retrieval.py pets 0 experiments/results/05_retrieval/pets_0.csv     # one-class retrieval
python experiments/06_sklearn_lp.py pets 0 experiments/results/06_sklearn_lp/pets_0.csv   # sklearn label propagation diagnosis

python experiments/make_report.py                   # rebuild this page from experiments/results/</code></pre>
<div class="tablewrap">{table(['File', 'What it is'], [
    ['<code>src/feature_spreading/run.py</code>',      'The original method: kNN graph, PPR, feature spreading, SVM.'],
    ['<code>src/feature_spreading/agop.py</code>',     'The first attempt: AGOP from the few-shot predictor (section 3).'],
    ['<code>experiments/01_agop_sweep.py</code>',      'Grid over the first attempt’s settings, plus the two ceilings. Pets, one seed, stopped early.'],
    ['<code>experiments/02_explore.py</code>',         'Stage 1: the wide search over ways to combine SVM, PPR and AGOP. Pets, 5 seeds.'],
    ['<code>experiments/03_explore2.py</code>',        'Stage 2: the two working variants on five datasets, with first controls.'],
    ['<code>experiments/04_ablate.py</code>',          'Stage 3: every number in sections 5 and 6.'],
    ['<code>experiments/05_retrieval.py</code>',       'Every number in section 8.'],
    ['<code>experiments/06_sklearn_lp.py</code>',      'Every number in section 7.'],
    ['<code>experiments/09_knn_variants.py</code>',    'The neighbor-count check in section 6.'],
    ['<code>experiments/results/</code>',              'One CSV per (experiment, dataset, seed).'],
], num_from=9)}</div>

<h2 id="glossary">12. Glossary</h2>
<dl class="gloss">
<dt>1-shot, 5-shot</dt><dd>One, or five, labeled examples per class. Everything else is unlabeled.</dd>
<dt>Transductive</dt><dd>The points to be labeled are all available while the method runs, so it can use their positions.</dd>
<dt>kNN graph</dt><dd>Each point is linked to its most similar points by cosine similarity, links made mutual. The code’s k = 10 gives 8 real neighbors.</dd>
<dt>Edge homophily</dt><dd>Fraction of graph edges joining two points of the same class. Needs all labels, so it is only a diagnostic.</dd>
<dt>PPR</dt><dd>Personalized PageRank. The weights a restart-prone random walk assigns to the points around its start.</dd>
<dt>α (alpha)</dt><dd>Probability the walk continues instead of restarting. Higher means smoothing reaches further. Default 0.85.</dd>
<dt>Feature spreading</dt><dd>Smooth the vectors with PPR, then train a normal classifier on the labeled rows.</dd>
<dt>Label propagation</dt><dd>Smooth a one-hot label matrix with PPR and take the largest entry per point.</dd>
<dt>AGOP</dt><dd>Average gradient outer product of a predictor: a matrix describing which input directions it is sensitive to.</dd>
<dt>RFM</dt><dd>Recursive feature machine. Alternates fitting a predictor and re-weighting the input space by its AGOP.</dd>
<dt>Ridge regression</dt><dd>Least-squares linear regression with a penalty on large weights. Solved in closed form.</dd>
<dt>Pseudo-labels</dt><dd>A classifier’s own predictions on unlabeled points, treated as if they were labels.</dd>
<dt>λ (lambda)</dt><dd>How much of the learned metric to mix with the plain one. 0 is plain cosine, 1 is the learned metric alone.</dd>
<dt>Average precision</dt><dd>For a ranked list, the mean of the precision measured at each true match. 1.0 is a perfect ranking.</dd>
<dt>Ceiling / oracle</dt><dd>A result computed using every true label, to show how much room is left. Never a usable method.</dd>
</dl>

</div>
<div id="tip" hidden></div>
"""

SECTIONS = {}

def build():
    body = BODY
    for k, v in SECTIONS.items():
        body = body.replace(k + '_PLACEHOLDER', v)
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           '<title>Feature Spreading Report</title>\n<style>' + CSS + '</style>\n</head>\n<body>\n' + body + '\n<script>' + JS + '</script>\n</body>\n</html>\n')
    path = os.path.join(HERE, 'report.html')
    open(path, 'w').write(doc)
    print(f'wrote {path} ({len(doc) / 1024:.0f} KB)')

# --
# Result sections (all numbers computed from the CSVs)

def names(lst):
    lst = list(lst)
    if not lst:          return 'none'
    if len(lst) == 1:    return lst[0]
    return ', '.join(lst[:-1]) + ' and ' + lst[-1]

def count_of(n):
    return f'all {n_ds}' if n == n_ds else f'{n} of {n_ds}'

GRAPH_KEYS = ['ppr', 'lp', 'km', 'km_lp', 'pl', 'pl_lp', 'plkm', 'plkm_lp']
NEW_KEYS   = ['km', 'km_lp', 'pl', 'pl_lp', 'plkm', 'plkm_lp']

def best_new(d, npc):
    return max(hv(k, d, npc) for k in NEW_KEYS)

svm_beats_ppr  = {n : [d for d in DS if hv('svc', d, n) > hv('ppr', d, n)] for n in NPCS}
svm_beats_all  = {n : [d for d in DS if hv('svc', d, n) > max(hv(k, d, n) for k in GRAPH_KEYS)] for n in NPCS}
km_loses       = {n : [d for d in DS if delta('km', d, n) < 0] for n in NPCS}
pl_loses       = {n : [d for d in DS if delta('pl', d, n) < 0] for n in NPCS}
lp_beats_all   = {n : [d for d in DS if max(hv('lp', d, n), hv('km_lp', d, n)) >= max(hv(k, d, n) for k in GRAPH_KEYS) - 1e-9 and hv('lp', d, n) > best_new(d, n)] for n in NPCS}
gap_to_ceiling = {n : float(np.nanmean([hv('oracle_agop', d, n) - best_new(d, n) for d in DS])) for n in NPCS}
std1           = {k : A[HEAD[k][1] & (A.npc == 1)].groupby('dataset').acc.std().reindex(DS) for k in ['ppr', 'km', 'pl']}

def _svm_note():
    parts = []
    for n in NPCS:
        a, b = svm_beats_ppr[n], svm_beats_all[n]
        if not a and not b:
            parts.append(f'At {n}-shot every graph method beats the plain SVM on every dataset.')
            continue
        t = f'At {n}-shot a plain SVM on the raw vectors beats the original feature spreading on {names(a)}'
        t += f', and beats every graph variant tested on {names(b)}.' if b else ', though the learned-graph variants recover the lead there.'
        parts.append(t)
    tail = ' These comparisons use the default graph settings; section 6 shows that lighter smoothing changes the picture at 5-shot.' if any(svm_beats_all.values()) else ''
    return '<p class="note">Graph methods are not always the right tool. ' + ' '.join(parts) + tail + '</p>'

svm_note = _svm_note()

def loss_list(key, npc):
    out = [f'{d} ({pts(delta(key, d, npc))})' for d in DS if delta(key, d, npc) < 0]
    return ', '.join(out) if out else 'none'

def mean_acc(key, npc):
    return float(np.nanmean([hv(key, d, npc) for d in DS]))

SECTIONS['RESULTS'] = f"""
<p>Every number is accuracy on the unlabeled points, averaged over seeds (each seed draws a different sample of points and a different labeled set). Settings were fixed in advance and are the same for every dataset: k = 10 (8 real neighbors), α = 0.85, K = 100 clusters, λ = 0.5, ridge penalty 1. Cluster targets use one round, pseudo-label targets use three. The final classifier is a linear SVM trained on the labeled points only.</p>
{FIG_HEAD}
<ul>
<li><strong>Cluster targets (no labels in the graph).</strong> {pts(mean_delta('km', 1))} points on average at 1-shot (range {rng_('km', 1)}), ahead of feature spreading by at least a point on {count_of(n_wins('km', 1, tol=0.01))} datasets and level on the rest. At 5-shot: {pts(mean_delta('km', 5))} on average (range {rng_('km', 5)}), ahead by at least half a point on {count_of(n_wins('km', 5, tol=0.005))}.</li>
<li><strong>Pseudo-label targets.</strong> {pts(mean_delta('pl', 1))} points on average at 1-shot (range {rng_('pl', 1)}) and {pts(mean_delta('pl', 5))} at 5-shot (range {rng_('pl', 5)}). Below feature spreading at 1-shot on: {loss_list('pl', 1)}. At 5-shot: {loss_list('pl', 5)}.</li>
<li><strong>Both targets together</strong> average {pts(mean_delta('plkm', 1))} at 1-shot and {pts(mean_delta('plkm', 5))} at 5-shot.</li>
<li><strong>Label propagation on the learned graph</strong> scores {pts(mean_acc('km_lp', 1) - mean_acc('km', 1))} points relative to the SVM on the same cluster-metric graph at 1-shot, and {pts(mean_acc('km_lp', 5) - mean_acc('km', 5))} at 5-shot, on average. It is the better final step when labels are scarcest.</li>
<li><strong>Remaining headroom.</strong> A metric built from all the true labels is still {100 * gap_to_ceiling[1]:.1f} points above the best real variant at 1-shot, averaged over datasets, and {100 * gap_to_ceiling[5]:.1f} at 5-shot.</li>
</ul>
{svm_note}
<p>Seed-to-seed variation at 1-shot is large on some datasets, because one labeled example per class can be a lucky or unlucky draw. The standard deviation of feature spreading across seeds ranges from {std1['ppr'].min():.3f} to {std1['ppr'].max():.3f} depending on the dataset; for the cluster-metric graph it is {std1['km'].min():.3f} to {std1['km'].max():.3f}. Differences of a point or less between variants should not be read as real.</p>

<h3>What the metric does to the graph</h3>
<p>The learned metric removes cross-class edges: edge homophily rises on every dataset, most where accuracy rises most, and barely at all on fashion_mnist, where accuracy does not move.</p>
<div class="tablewrap">{hom_table()}</div>
<p>Homophily is not the whole explanation. The control that rebuilds the graph on smoothed features with no metric reaches similar homophily on several datasets, yet gains less than half as much accuracy (section 6). Which edges are kept matters, not only how many of them join the same class. That difference was not investigated further.</p>
"""

# -- ablations

def kmean(K, npc, pred='svc'):
    return float(np.nanmean([k_get(npc, d)(pred, K) for d in DS]))

BEST_K   = {n : max(K_GRID, key=lambda K: kmean(K, n)) for n in NPCS}
K_PER_DS = {d : max(K_GRID, key=lambda K: k_get(1, d)('svc', K)) for d in DS}
K_SAFE   = [K for K in K_GRID if all(k_get(n, d)('svc', K) > hv('ppr', d, n) - 0.005 for d in DS for n in NPCS)]
K50_LOSES = [d for d in DS if k_get(1, d)('svc', 50) < hv('ppr', d, 1) - 0.005]

def sweep_mean(col, v, sweep, default, npc, it=1):
    m = m_km('svc', it) if v == default else m_km('svc', it, sweep, **{col : v})
    return float(np.nanmean(acc(m, npc).values))

def ka(k, alpha, npc, km=True, ds=None):
    m = ((m_km('svc', 1) if (k, alpha) == (10, 0.85) else m_km('svc', 1, 'k_alpha', k=k, alpha=alpha)) if km else m_ppr('svc', k, alpha))
    a = acc(m, npc)
    return float(a[ds]) if ds else float(np.nanmean(a.values))

def _kv(graph, k, w):
    m = KV[(KV.graph == graph) & (KV.k == k) & (KV.self_w == w)].copy()
    m['col'] = m.task + m.npc.astype(str) + m.pred
    return m.groupby(['col', 'dataset']).score.mean().unstack('dataset').reindex(columns=DS)

KV_COLS = [('classify1svc', '1-shot, SVM'), ('classify1lp', '1-shot, label prop.'), ('classify5svc', '5-shot, SVM'), ('classify5lp', '5-shot, label prop.'), ('retrieve1ppr', 'retrieval, 1 exemplar')]
def kv_table():
    rows = []
    for g, gname in [('knn', 'Plain graph'), ('cluster_metric', 'Cluster-metric graph')]:
        for k, w, name in [(8, 2, '8 neighbors, self-loop ×2 (the code)'), (8, 1, '8 neighbors'), (10, 1, 'clean 10 neighbors'), (10, 2, '10 neighbors, self-loop ×2')]:
            t = _kv(g, k, w)
            rows.append([f'{gname}: {name}'] + [float(t.loc[c].mean()) for c, _ in KV_COLS])
    return table(['Graph'] + [n for _, n in KV_COLS], rows, caption='Mean over six datasets. Accuracy for classification, mean average precision for retrieval.')

kv_max  = max(abs(float(_kv(g, 8, 2).loc[c].mean() - _kv(g, 10, 1).loc[c].mean())) for g in ['knn', 'cluster_metric'] for c, _ in KV_COLS)
kv_cars = float((_kv('cluster_metric', 8, 2)['cars'] - _kv('cluster_metric', 10, 1)['cars']).mean())

def abl_val(label_start, npc):
    return [r for r in abl_rows(npc) if r['label'].startswith(label_start)][0]['value']

SECTIONS['ABLATIONS'] = f"""
<p>Each test changes one thing from the default and keeps the rest fixed. The default is k-means with K = 100 on the smoothed features, λ = 0.5, ridge penalty 1, one round, SVM on the labeled points.</p>

<h3>How many clusters</h3>
<p>The number of clusters does not need to match the number of classes, and the method does not need to know the class count. What matters is having enough clusters. A ridge model with K outputs produces a metric with at most K − 1 directions, so a small K squeezes the graph into a space too small to hold the class structure.</p>
{FIG_K}
<ul>
<li>K = 10 is harmful: mean 1-shot accuracy {p(kmean(10, 1))}, against {p(mean_acc('ppr', 1))} for feature spreading with no metric at all.</li>
<li>The best single setting on average is K = {BEST_K[1]} at 1-shot ({p(kmean(BEST_K[1], 1))}) and K = {BEST_K[5]} at 5-shot ({p(kmean(BEST_K[5], 5))}). Beyond that, accuracy falls off slowly: K = 400 averages {p(kmean(400, 1))} at 1-shot.</li>
<li>K = {names(str(K) for K in K_SAFE)} are level with or ahead of feature spreading on every dataset at both shot counts. K = 50 is not: at 1-shot it falls short on {names(K50_LOSES)}, the two datasets with about 100 classes.</li>
<li>Best K per dataset at 1-shot: {', '.join(f'{d} {K_PER_DS[d]}' for d in DS)}. The link to the class count is loose (cars has 196 classes and is best at 100; fashion_mnist has 10 and is best at 400), so the practical rule is to err high.</li>
</ul>
<p>An earlier run set K equal to the class count. That was the cause of a loss on fashion_mnist, which has 10 classes: with K = 10 the cluster-metric graph scores {p(k_get(1, 'fashion_mnist')('svc', 10))} there at 1-shot, and with K = 100 it scores {p(hv('km', 'fashion_mnist', 1))}, level with feature spreading ({p(hv('ppr', 'fashion_mnist', 1))}). The metric no longer hurts on that dataset, and it does not help either.</p>

<h3>Metric strength, ridge penalty, rounds</h3>
<ul>
<li><strong>λ (how much of the learned metric to use).</strong> Mean 1-shot accuracy is {p(sweep_mean('lam', 0.1, 'lam', 0.5, 1))} at λ = 0.1, {p(sweep_mean('lam', 0.5, 'lam', 0.5, 1))} at 0.5 and {p(sweep_mean('lam', 1.0, 'lam', 0.5, 1))} at 1.0. More is slightly better and nothing breaks at 1.0, where the plain metric is dropped entirely.</li>
<li><strong>Ridge penalty.</strong> {p(sweep_mean('reg', 0.01, 'reg', 1.0, 1))} at 0.01, {p(sweep_mean('reg', 0.1, 'reg', 1.0, 1))} at 0.1, {p(sweep_mean('reg', 1.0, 'reg', 1.0, 1))} at 1 and {p(sweep_mean('reg', 10.0, 'reg', 1.0, 1))} at 10 (mean 1-shot accuracy). The default of 1 is the best on average. A penalty of 0.01 costs {100 * (sweep_mean('reg', 1.0, 'reg', 1.0, 1) - sweep_mean('reg', 0.01, 'reg', 1.0, 1)):.1f} points and a penalty of 10 costs {100 * (sweep_mean('reg', 1.0, 'reg', 1.0, 1) - sweep_mean('reg', 10.0, 'reg', 1.0, 1)):.1f}.</li>
<li><strong>Rounds.</strong> One round gets nearly everything. With cluster targets, mean 1-shot accuracy is {p(float(np.nanmean(acc(m_km('svc', 1), 1).values)))} after one round and {p(float(np.nanmean(acc(m_km('svc', 4), 1).values)))} after four. With pseudo-label targets it is {p(float(np.nanmean(acc(m_simple('pl_agop', 'svc', 1), 1).values)))} after one and {p(float(np.nanmean(acc(m_simple('pl_agop', 'svc', 3), 1).values)))} after three. Pets is the exception: pseudo-label rounds take it from {p(float(acc(m_simple('pl_agop', 'svc', 1), 1)['pets']))} to {p(float(acc(m_simple('pl_agop', 'svc', 3), 1)['pets']))}.</li>
</ul>
<details class="numbers"><summary>Show the λ, ridge and rounds tables</summary>
{sweep_table('lam', [0.1, 0.25, 0.5, 0.75, 1.0], 'lam', 0.5)}
{sweep_table('reg', [0.01, 0.1, 1.0, 10.0], 'reg', 1.0)}
{rounds_table()}
</details>

<h3>Which parts are doing the work</h3>
{FIG_ABL}
<ul>
<li><strong>The targets need real structure.</strong> A metric fit to random targets makes things worse ({pts(abl_val('Control: random', 1))} points on average). The gain is not a side effect of passing the data through a ridge regression.</li>
<li><strong>It is not a principal-component effect.</strong> Keeping the top 100 principal directions of the smoothed features changes nothing ({pts(abl_val('Control: project', 1))}), and whitening them is harmful ({pts(abl_val('Control: whiten', 1))}). The ridge model picks directions that separate clusters, which is different from picking directions with high variance.</li>
<li><strong>Re-graphing alone gets part of the way.</strong> Rebuilding the neighbor graph on the smoothed features, with no metric at all, gives {pts(abl_val('Control: re-graph', 1))} points on average. The cluster metric gives {pts(abl_val('Default', 1))}. So part of the benefit is simply “build the graph twice”, and the metric adds the rest.</li>
<li><strong>Smoothing before clustering is optional.</strong> Running k-means and the ridge fit on the raw vectors gives {pts(abl_val('Cluster and fit on raw', 1))}, about the same as the default. The simplest version of the method is then: k-means on the raw vectors, ridge regression, metric, graph, spread.</li>
<li><strong>Build the new graph from the raw vectors.</strong> Applying the metric to the smoothed features and building the graph from those is worse ({pts(abl_val('Build the new graph', 1))}).</li>
</ul>

<h3>Neighbors and restart probability</h3>
<p>On average the learned metric helps at every graph setting tried: it is ahead of feature spreading at the same k and α in all five combinations, at both shot counts.</p>
<p>The best amount of smoothing depends on how many labels there are. With one label per class, heavy smoothing is right: α = 0.85 averages {p(ka(10, 0.85, 1))} with the cluster-metric graph, against {p(ka(10, 0.5, 1))} at α = 0.5. With five labels per class, lighter smoothing is better: α = 0.5 averages {p(ka(10, 0.5, 5))} and k = 5 averages {p(ka(5, 0.85, 5))}, against {p(ka(10, 0.85, 5))} at the default. The difference is largest on cars, where the default cluster-metric graph scores {p(ka(10, 0.85, 5, ds='cars'))} at 5-shot, below the plain SVM ({p(hv('svc', 'cars', 5))}), and α = 0.5 scores {p(ka(10, 0.5, 5, ds='cars'))}, above it. Pushing α up to 0.95 lowers the average at both shot counts, with or without the metric.</p>
<details class="numbers"><summary>Show the k and α table</summary>{kalpha_table()}</details>

<h3>The neighbor-count quirk</h3>
<p>Since the code’s k = 10 is really 8 neighbors with a doubled self-loop, a clean 10-nearest-neighbor graph with a single self-loop was run for comparison, with the k-means and ridge fit on the raw vectors. Averaged over datasets it is within {100 * kv_max:.1f} points of the original on every measure. The self-loop weight makes no measurable difference. The neighbor count matters slightly on cars, the dataset with the weakest graph, where the clean 10 is about {100 * kv_cars:.1f} points lower.</p>
<div class="tablewrap">{kv_table()}</div>

<h3>Kernel or linear AGOP</h3>
<p>The original idea used a kernel predictor, whose AGOP has to be computed by averaging gradients over every point. The linear version is a single matrix product. Where the kernel version was affordable (50 classes or fewer), the two are close: kernel is up to a point and a half ahead at 5-shot and about the same at 1-shot, at many times the cost.</p>
<div class="tablewrap">{kernel_table()}</div>

<h3>Things that did not help</h3>
<p>The first wide search, on pets only, tried other ways to use the same ingredients. These are the ones that failed, next to the one that worked:</p>
<div class="tablewrap">{E1_TABLE}</div>
<ul>
<li><strong>Unsupervised graph tweaks</strong> (centering the vectors, changing k or α, weighting edges by similarity) move accuracy by about a point, and that is after picking the best of 72 settings using the test labels.</li>
<li><strong>Self-training</strong> without changing the graph does not help, and gets worse with more rounds as wrong pseudo-labels reinforce themselves.</li>
<li><strong>Deleting edges</strong> between points with different predicted labels is the direct imitation of the “no cross-class edges” ceiling. It does nothing, because the edges it removes are the ones the classifier already agreed with.</li>
<li><strong>A class-mean metric</strong> (directions between the pseudo-class averages, no regression) helps a little at 1-shot and not at all at 5-shot. The ridge fit matters: it also accounts for how the data is spread within classes.</li>
<li><strong>Applying the metric to the features</strong> the classifier sees, instead of to the graph, is clearly worse. Applying it to both is no better than the graph alone.</li>
<li><strong>Training the final classifier on pseudo-labels</strong> is worse than training it on the few true labels over the improved features.</li>
</ul>
"""

# -- label propagation

lp_never_lo, lp_never_hi = float(LP_NEVER.min()), float(LP_NEVER.max())
_sklp     = L[L.method == 'sklearn LabelPropagation (directed kNN)']
lp_runs   = len(_sklp)
lp_capped = int((_sklp.n_iter >= 2000).sum())
SECTIONS['LP'] = f"""
<p>The repository was written partly because label propagation “basically didn’t work” on this kind of data. That observation was about scikit-learn’s implementation, and it still holds in the current version (1.9.1). It is a property of the graph that implementation builds, not of label propagation as an idea.</p>
{FIG_LP}
<p>With <code>kernel='knn'</code>, scikit-learn builds a <em>directed</em> neighbor graph: row <em>i</em> lists the points that <em>i</em> considers its nearest neighbors, and the update step has each point pull label mass from its own list. Information therefore flows from a labeled point only to points that list it. In high-dimensional data a noticeable share of points appear in nobody’s list. Here that is {100 * lp_never_lo:.0f}% to {100 * lp_never_hi:.0f}% of the labeled points, depending on the dataset. When the single labeled example of a class is one of those, its label never leaves it, and the whole class is unreachable. That is the extreme case. More generally, how far a label travels depends on how many points happen to list the labeled one, and in high dimensions that count is very uneven: a few points are in hundreds of lists and many are in almost none.</p>
<p>The test is to swap in the symmetrized graph (two points are linked if either lists the other) and change nothing else:</p>
<ul>
<li><code>LabelSpreading</code> as shipped averages {p(mean_(LP_SK[1]))} at 1-shot. With the symmetrized graph it averages {p(mean_(LP_SYM[1]))}, level with PPR on one-hot labels over this repository’s graph ({p(mean_(LP_PPR[1]))}). At 5-shot: {p(mean_(LP_SK[5]))}, {p(mean_(LP_SYM[5]))} and {p(mean_(LP_PPR[5]))}.</li>
<li><code>LabelPropagation</code> as shipped is worse still ({p(mean_(LP_SKLP[1]))} at 1-shot). It also clamps the labeled points hard and iterates until convergence, and it hit the 2,000-iteration limit on {lp_capped} of its {lp_runs} runs.</li>
</ul>
<p>So label propagation is a reasonable baseline once the graph is right. On the base graph it is close to feature spreading (mean accuracy {p(mean_acc('lp', 1))} against {p(mean_acc('ppr', 1))} at 1-shot, {p(mean_acc('lp', 5))} against {p(mean_acc('ppr', 5))} at 5-shot), ahead on some datasets and behind on others. Whether scikit-learn’s behavior changed between versions was not checked; the directed graph is enough to explain what was seen.</p>
"""

# -- retrieval

ret_keys_real = [k for k in RET if not k.startswith('or_')]
_Q = R[R.subset == 'q20']
def qd(lam, npc):
    a = _Q[(_Q.graph == 'query_agop') & (_Q.scorer == 'ppr') & (_Q.lam == lam) & (_Q.npc == npc)].groupby('dataset')['map'].mean().reindex(DS)
    b = _Q[(_Q.graph == 'knn') & (_Q.scorer == 'ppr') & (_Q.npc == npc)].groupby('dataset')['map'].mean().reindex(DS)
    return float(np.nanmean((a - b).values))
def rmean_g(graph, npc):   # PageRank at alpha = 0.85, all queries
    kw = dict(k=10) if graph == 'knn' else KM
    return float(np.nanmean(rmap(graph, 'ppr', npc, ppr_alpha=0.85, **kw).values))
RBEST = {n : max(ret_keys_real, key=lambda k: rmean(k, n)) for n in NPCS}
km_ppr_wins = {n : sum(rv('km_ppr', d, n) > rv('knn_ppr', d, n) for d in DS) for n in NPCS}

SECTIONS['RETRIEVAL'] = f"""
<p>A different question from classification: you have one example (or five) of a <em>single</em> class and want to find the rest of that class in the pile. There are no labels for any other class, no negatives, and the number of classes is unknown. The output is a ranking of every other point.</p>
<p>Two measures are reported. <em>Average precision</em> walks down the ranking and averages the precision at each true match, so it rewards putting all members of the class near the top; its mean over queries is the headline number. <em>Precision in the top 10</em> is the fraction of the first ten results that are correct. Each (dataset, seed) uses at least 100 queries, with exemplars drawn at random from each class.</p>
<p>The scorers:</p>
<ul>
<li><strong>Cosine similarity</strong> to the exemplar. With one positive and nothing else, this is what a linear classifier reduces to.</li>
<li><strong>One-vs-rest classifier</strong>: the exemplar as the positive, unlabeled points as negatives. Tried as a linear SVM against 500 random points and as a closed-form ridge fit against all points.</li>
<li><strong>Query expansion</strong>: add the top 10 results to the query and search again. A standard retrieval trick, included as a non-graph baseline.</li>
<li><strong>PageRank from the exemplar</strong>: the PPR weights the exemplar assigns to every other point. This is label propagation with a single class.</li>
</ul>
<p>Each scorer can run on the raw vectors or on features spread over a graph, and the graph can be the plain neighbor graph or the cluster-metric graph from section 4. The cluster-metric graph uses no labels, so it is built once and shared by every query.</p>
{FIG_RET}
<ul>
<li><strong>Spreading helps retrieval a great deal.</strong> Mean average precision with one exemplar goes from {p(rmean('cos', 1))} for cosine on raw vectors to {p(rmean('knn_cos', 1))} for cosine on spread features, averaged over datasets.</li>
<li><strong>The cluster-metric graph adds to that.</strong> PageRank on it averages {p(rmean('km_ppr', 1))}, against {p(rmean('knn_ppr', 1))} on the plain graph, and is ahead on {count_of(km_ppr_wins[1])} datasets. With five exemplars: {p(rmean('km_ppr', 5))} against {p(rmean('knn_ppr', 5))}, ahead on {count_of(km_ppr_wins[5])}.</li>
<li><strong>Best scorer.</strong> With one exemplar: “{RET[RBEST[1]][0]}” ({p(rmean(RBEST[1], 1))}; one round gives {p(rmean('km_ppr', 1))}). With five: “{RET[RBEST[5]][0]}” ({p(rmean(RBEST[5], 5))}).</li>
<li><strong>A one-vs-rest classifier needs spread features, or more than one exemplar.</strong> With one exemplar on raw vectors the SVM averages {p(rmean('svc', 1))} and the ridge version {p(rmean('elda', 1))}, against {p(rmean('cos', 1))} for plain cosine: little or no gain. Trained on features spread over the cluster-metric graph, the same SVM reaches {p(rmean('km_svc', 1))}. With five exemplars the classifier earns its place: {p(rmean('svc', 5))} on raw vectors against {p(rmean('cos', 5))} for cosine, and {p(rmean('km_svc', 5))} (SVM) or {p(rmean('km_elda', 5))} (ridge) on the cluster-metric graph.</li>
<li><strong>The metric needs the graph.</strong> Cosine similarity under the learned metric, with no graph or spreading, averages {p(rmean('metric_cos', 1))}. That is better than raw cosine and well short of the graph versions.</li>
<li><strong>Query expansion</strong> on raw vectors reaches {p(rmean('aqe', 1))}: a small gain for a small change, far below spreading.</li>
<li><strong>The top of the ranking moves less.</strong> Precision in the top 10 with one exemplar is {p(rmean('cos', 1, 'p10'))} for raw cosine, {p(rmean('knn_cos', 1, 'p10'))} on spread features and {p(rmean('km_cos', 1, 'p10'))} on the cluster-metric graph. The nearest few neighbors are mostly right already; the graph mainly helps find the members of the class that are far from the exemplar.</li>
<li><strong>Headroom.</strong> The all-labels ceiling is {p(rmean('or_ppr', 1))} with one exemplar and {p(rmean('or_ppr', 5))} with five.</li>
</ul>

<h3>Retrieval settings</h3>
<p>The number of clusters behaves as it does in classification: too few hurts, K = 50 is not enough for the datasets with about 100 classes, and 100 to 400 is safe. For the PageRank scorer, a restart setting of α = 0.95 is the best on average on both graphs.</p>
<details class="numbers"><summary>Show the cluster-count and α tables</summary>{ret_k_table()}{ret_alpha_table()}</details>

<h3>A metric built for each query</h3>
<p>The retrieval analogue of pseudo-labels is to treat the exemplar’s top 20 neighbors as positives, fit a one-vs-rest ridge model, and use its AGOP (a single direction) as a per-query metric. This needs a new graph and a new solve for every query, so it was only run on 20 queries per seed.</p>
<div class="tablewrap">{ret_query_table()}</div>
<p>With a light touch (λ = 0.02) it helps: {pts(qd(0.02, 1))} points of average precision with one exemplar and {pts(qd(0.02, 5))} with five, relative to PageRank on the plain graph for the same queries. For comparison, the shared cluster-metric graph gives {pts(rmean_g('kmeans_agop', 1) - rmean_g('knn', 1))} and {pts(rmean_g('kmeans_agop', 5) - rmean_g('knn', 5))} over the same baseline on all queries, at no per-query cost. So the per-query metric is only competitive with five exemplars, and it needs a new graph for every query. Combining the two was not tried.</p>
"""

SECTIONS['RECIPE'] = f"""
<p>For few-shot classification or one-class search over a fixed set of generic vectors:</p>
<pre><code>X      = normalize(X)                                   # rows to unit length
labels = kmeans(X, K=100)                               # K = 100–400; too few clusters is harmful
                                                        # (clustering raw X scored the same as clustering spread X)
W      = ridge(X, one_hot(labels), penalty=n / 5000)    # one linear solve; W is K × d. Penalty 1 was tuned at
                                                        # n ≈ 5,000; scale it with n to keep the same strength
G      = W.T @ W                                        # AGOP of the linear model
M      = (1 - lam) * I + lam * G * d / trace(G)         # lam = 0.5 (anything from 0.5 to 1 is fine)
adj    = knn_graph(normalize(X @ sqrtm(M)), k=8)        # 8 real neighbors (what the code's k=10 does); 10 is
                                                        # within about half a point. Mutual: linked if either lists the other
P      = normalize(ppr(adj, alpha=0.85) @ X)            # spread the ORIGINAL vectors

# classification, few labels per class
clf    = LinearSVC().fit(P[labeled], y[labeled])        # or: argmax of ppr(adj, alpha=0.95) @ one_hot(y_labeled)

# retrieval, one class
scores = ppr(adj, alpha=0.95) @ indicator(exemplars)    # rank by score</code></pre>
<ul>
<li><strong>With 1 label per class,</strong> prefer label propagation on the learned graph as the final step. With 5, the SVM and label propagation are close.</li>
<li><strong>With 5 labels per class, smooth less.</strong> α = 0.5 (or k = 5) is better than the default by about a point on average and by several points where the neighbor graph is poor.</li>
<li><strong>Pseudo-labels are an optional extra.</strong> Adding the classifier’s predictions as targets next to the clusters, for three rounds, is worth {pts(mean_delta('plkm', 1) - mean_delta('km', 1))} points at 1-shot and {pts(mean_delta('plkm', 5) - mean_delta('km', 5))} at 5-shot on average, and costs a classifier fit per round.</li>
<li><strong>For retrieval with several exemplars,</strong> a one-vs-rest classifier trained on <code>P</code> is as good as or better than PageRank. With one exemplar, use PageRank.</li>
<li><strong>Check against a plain SVM on the raw vectors.</strong> When the base neighbor graph is poor, spreading of any kind can lose to it, and there is no label-free test here that predicts when.</li>
<li><strong>Cost.</strong> k-means, one d × d solve, one extra neighbor search and one extra PPR solve. In these experiments PPR is a dense n × n solve, which is fine at 5,000 points and would need a sparse or iterative solver beyond a few tens of thousands.</li>
</ul>
"""

SECTIONS['CAVEATS'] = f"""
<ul>
<li><strong>Few seeds.</strong> {seeds_note}. One-shot accuracy varies by several points between seeds on some datasets. Averages over datasets are more trustworthy than any single cell.</li>
<li><strong>Settings were chosen with test labels in view.</strong> There is no validation set at 1-shot. The defaults (K = 100, λ = 0.5, ridge 1) were fixed as the center of the sweeps before the cross-dataset runs, and the sweeps show accuracy is flat around them, but the earlier exploration that led to this design used pets heavily. Pets is also where the gains are largest.</li>
<li><strong>One kind of vector.</strong> All six datasets are embeddings from the same image model. The methods are generic, but the results have not been checked on embeddings from other models, on text, or on tabular data.</li>
<li><strong>Subsampled.</strong> Each run uses at most 5,000 points. Neighbor graphs get denser and usually better with more points, which could shrink or grow these gaps.</li>
<li><strong>Settings tuned at one size.</strong> The ridge penalty of 1 is not scaled by the number of points. The quantity it is added to grows in proportion to n, so on a much larger collection the same penalty is relatively weaker; scaling it as n / 5,000 keeps the tested strength. The cluster count (100) and neighbor count were likewise only tested at 3,700 to 5,000 points.</li>
<li><strong>Transductive only.</strong> Every method here labels the points it was given. Labeling a new point later would need it to be added to the graph.</li>
<li><strong>Retrieval negatives.</strong> The one-vs-rest scorers treat unlabeled points as negatives, some of which are true positives. That matches the real setting and is part of why those scorers do poorly with one exemplar.</li>
<li><strong>The ceilings are not targets.</strong> They use all labels to build the graph. They show the graph is still the limiting factor, not what a label-free method could reach.</li>
</ul>
"""

summary_warn = (f'At 5-shot with default settings a plain SVM on the raw vectors beats every graph variant on {names(svm_beats_all[5])}; lighter smoothing is needed there.' if svm_beats_all[5]
                else f'At 5-shot a plain SVM on the raw vectors beats the original feature spreading on {names(svm_beats_ppr[5])}.' if svm_beats_ppr[5]
                else 'The gains are uneven across datasets, and small on some.')
lp_fix = mean_(LP_SYM[1]) - mean_(LP_SK[1])
SECTIONS['SUMMARY'] = f"""
<ul class="findings">
<li><strong>A learned metric improves the graph, if it is learned from every point.</strong> Cluster the points, fit a linear model to the cluster assignments, and use its AGOP as the metric for the neighbor graph. With no labels in the graph, accuracy over feature spreading changes by {pts(mean_delta('km', 1))} points at 1-shot and {pts(mean_delta('km', 5))} at 5-shot on average. It is ahead by at least a point on {count_of(n_wins('km', 1, tol=0.01))} datasets at 1-shot and never behind.</li>
<li><strong>The original AGOP loop stalled for a specific reason.</strong> A predictor fit on one example per class only describes those examples. Giving it a target for every point (clusters or pseudo-labels) is the whole fix.</li>
<li><strong>Use enough clusters.</strong> K = {K_SAFE[0]} to {K_SAFE[-1]} is level with or ahead of feature spreading on every dataset, whatever the class count. Too few is actively harmful.</li>
<li><strong>The same graph helps one-class retrieval.</strong> With one exemplar, mean average precision goes from {p(rmean('cos', 1))} (cosine on raw vectors) to {p(rmean('knn_cos', 1))} (cosine on spread features) to {p(rmean('km_ppr', 1))} (PageRank on the cluster-metric graph).</li>
<li><strong>Label propagation works once the graph is symmetric.</strong> scikit-learn’s kNN version uses a directed graph and scores {p(mean_(LP_SK[1]))} at 1-shot; symmetrizing it gives {p(mean_(LP_SYM[1]))}, the same as PPR on this repository’s graph.</li>
<li class="warn"><strong>Graph methods can still lose.</strong> {summary_warn} The gain is close to zero on fashion_mnist. The all-labels ceiling shows {100 * gap_to_ceiling[1]:.0f} points of headroom at 1-shot that none of this reaches.</li>
</ul>
"""


if __name__ == '__main__':
    build()
