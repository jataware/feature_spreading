"""
  table.py - collect blog/results/{baselines,ours}/*.csv into a readable results page

  usage (from repo root): python blog/table.py

  Writes blog/results/summary.html (open it in a browser) and prints a short text summary.

  Each method has a small grid. For every (method, shot count) the grid setting with the best accuracy averaged over
  the DEV datasets is used everywhere (each dataset is averaged over seeds first). Other datasets never influence a setting.
"""

import os
import sys
import glob
import html
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data

HERE = os.path.dirname(os.path.abspath(__file__))

SECTIONS = [
    ('base', 'Baselines · scikit-learn, labeled points only',  ['svm', 'logreg', 'nearest_centroid', 'knn']),
    ('base', 'Baselines · scikit-learn, semi-supervised',      ['label_propagation', 'label_spreading', 'self_training', 'kmeans_seeded']),
    ('base', 'Baselines · graphlearning',                      ['laplace', 'poisson', 'poisson_connected', 'randomwalk', 'graph_nn', 'centered_kernel']),
    ('ours', 'Ours · plain kNN graph',                         ['ls', 'fs', 'fs+ls']),
    ('ours', 'Ours · kNN graph under the cluster metric (label-free)', ['cmg_ls', 'cmg_fs', 'cmg_fs+ls']),
    ('ours', 'Ours · graph metric from clusters + pseudo-labels (uses the labels)', ['plg_ls', 'plg_fs', 'plg_fs+ls']),
    ('ceil', 'Ceilings · built with every true label',         ['oracle_metric', 'oracle_graph']),
]
NAMES = {
    'svm' : 'LinearSVC', 'logreg' : 'LogisticRegression', 'nearest_centroid' : 'NearestCentroid', 'knn' : 'KNeighborsClassifier (k=1)',
    'label_propagation' : 'LabelPropagation', 'label_spreading' : 'LabelSpreading', 'self_training' : 'SelfTrainingClassifier',
    'kmeans_seeded' : 'KMeans, seeded with class means',
    'laplace' : 'Laplace learning', 'poisson' : 'Poisson learning', 'poisson_connected' : 'Poisson learning, connected graph', 'randomwalk' : 'Random walk', 'graph_nn' : 'Graph nearest neighbor', 'centered_kernel' : 'Centered kernel',
    'ls' : 'spread the labels', 'fs' : 'spread the vectors + SVM', 'fs+ls' : 'both, scores added',
    'cmg_ls' : 'spread the labels', 'cmg_fs' : 'spread the vectors + SVM', 'cmg_fs+ls' : 'both, scores added',
    'plg_ls' : 'spread the labels', 'plg_fs' : 'spread the vectors + SVM', 'plg_fs+ls' : 'both, scores added',
    'oracle_metric' : 'metric from all true labels', 'oracle_graph' : 'graph with no cross-class edges',
}
ENC = {data.CLIP : 'CLIP', data.DINO : 'DINOv2', data.QWEN : 'Qwen3', data.MINILM : 'MiniLM', data.GEMINI : 'Gemini', data.CLAP : 'CLAP', 'raw' : 'raw'}

# --
# Load, keep seeds that every kind of run has for a dataset, choose settings on dev

d = pd.concat([pd.read_csv(f) for f in glob.glob(f'{HERE}/results/baselines*/*.csv') + glob.glob(f'{HERE}/results/ours*/*.csv')], ignore_index=True)
d = d[d.dataset.isin(data.REGISTRY)].copy()
d['variant'] = d.variant.fillna('')
d['group']   = d.dataset.map(lambda x: data.REGISTRY[x]['group'])

ok = d.groupby(['dataset', 'kind']).seed.agg(set).groupby('dataset').agg(lambda s: set.intersection(*s))
d  = d[[seed in ok[ds] for ds, seed in zip(d.dataset, d.seed)]]

datasets = [k for k in data.REGISTRY if k in set(d.dataset)]
groups   = {g : [x for x in datasets if data.REGISTRY[x]['group'] == g] for g in ['dev', 'other']}
kinds    = d.groupby('dataset').kind.agg(set)
shots    = sorted(d.shots.unique())
methods  = [m for _, _, ms in SECTIONS for m in ms if m in set(d.method)]

dev    = d[d.group == 'dev']
score  = dev.groupby(['method', 'shots', 'variant', 'dataset']).acc.mean().groupby(['method', 'shots', 'variant']).mean()
chosen = {k : v[2] for k, v in score.groupby(level=['method', 'shots']).idxmax().items()}
keep   = d[[chosen.get((m, s)) == v for m, s, v in zip(d.method, d.shots, d.variant)]]
cell   = keep.groupby(['shots', 'method', 'dataset']).acc.mean()       # mean over seeds

def val(s, m, ds):
    return cell.get((s, m, ds), np.nan)

def group_mean(s, m, g, need=None):
    """ mean over the datasets of a group that have results for `need` kinds of run (so rows stay comparable) """
    cols = [x for x in groups[g] if (need or {'baselines', 'ours'}) <= kinds[x]]
    vals = [val(s, m, x) for x in cols]
    return (np.mean(vals), len(cols)) if cols and not np.isnan(vals).any() else (np.nan, len(cols))

real_methods = [m for m in methods if not m.startswith('oracle')]
def ranks(s):
    """ rank of every non-ceiling method within each dataset (1 = best, ties share the average rank) """
    acc = cell.xs(s, level='shots').unstack('dataset').reindex(real_methods)
    return acc.rank(ascending=False, method='average')
RANK = {s : ranks(s) for s in shots}

def group_rank(s, m, g):
    cols = [x for x in groups[g] if {'baselines', 'ours'} <= kinds[x]]
    if m not in RANK[s].index or not cols:
        return np.nan
    return RANK[s].loc[m, cols].mean()

def setting(m, s):
    v = chosen.get((m, s), '')
    for junk in ['graph=plain,', 'graph=cmg,', 'graph=plg,', 'which=graph', 'which=metric']:
        v = v.replace(junk, '')
    return v.replace('alpha_ls', 'α labels').replace('alpha_fs', 'α vectors').replace('alpha', 'α').replace(',', ', ') or 'default'

# --
# HTML

def pct(v):
    return '–' if np.isnan(v) else f'{100 * v:.1f}'

def table(header_rows, body_rows, cls=''):
    return f'<div class="wrap"><table class="{cls}"><thead>{"".join(header_rows)}</thead><tbody>{"".join(body_rows)}</tbody></table></div>'

def best_of(values_by_method):
    real = {m : v for m, v in values_by_method.items() if not m.startswith('oracle') and not np.isnan(v)}
    return max(real.values()) if real else np.nan

def is_rank(c):
    return 'rank' in c

def body(columns, getter):
    """ columns: list of keys; getter(method, key) -> value. Rows = methods, grouped into sections. """
    best = {}
    for c in columns:
        vals = {m : getter(m, c) for m in methods if not m.startswith('oracle')}
        vals = [v for v in vals.values() if not np.isnan(v)]
        best[c] = (min(vals) if is_rank(c) else max(vals)) if vals else np.nan
    rows = []
    for kind, title, ms in SECTIONS:
        ms = [m for m in ms if m in methods]
        if not ms:
            continue
        rows.append(f'<tr class="sec {kind}"><th colspan="{len(columns) + 1}">{html.escape(title)}</th></tr>')
        for m in ms:
            tds = []
            for c in columns:
                v   = getter(m, c)
                top = (not m.startswith('oracle')) and not np.isnan(v) and abs(v - best[c]) < 5e-4
                txt = ('–' if np.isnan(v) else f'{v:.1f}') if is_rank(c) else pct(v)
                cls = ' '.join(filter(None, ['best' if top else '', 'gm' if c[0] == 'mean' else '', 'rk' if is_rank(c) else '']))
                tds.append(f'<td class="{cls}">{txt}</td>')
            rows.append(f'<tr class="{kind}"><th>{html.escape(NAMES[m])}</th>{"".join(tds)}</tr>')
    return rows

# summary: group means
both = {'baselines', 'ours'}
gs       = [(g, s) for g in ['dev', 'other'] for s in shots if groups[g]]
sum_cols = [(g, s, what) for g, s in gs for what in ['acc', 'rank']]
n_cmp    = {g : len([x for x in groups[g] if both <= kinds[x]]) for g in groups}
def sum_get(m, c):
    return group_mean(c[1], m, c[0])[0] if c[2] == 'acc' else group_rank(c[1], m, c[0])
glabel = {'dev' : 'Dev datasets', 'other' : 'Other datasets'}
sum_head = ['<tr><th rowspan="3">method</th>' + ''.join(f'<th colspan="{2 * len(shots)}" class="grp">{glabel[g]} <span class="dim">({n_cmp[g]})</span></th>' for g in ['dev', 'other'] if groups[g]) + '</tr>',
            '<tr>' + ''.join(f'<th colspan="2" class="grp">{s}-shot</th>' for g, s in gs) + '</tr>',
            '<tr>' + ''.join('<th>accuracy</th><th class="rk">mean rank</th>' for _ in gs) + '</tr>']
SUMMARY  = table(sum_head, body(sum_cols, sum_get), 'summary')

# by vector source: which encoder (or none) produced the vectors
FAMILY = {data.CLIP : 'CLIP images (dev)', data.DINO : 'DINOv2, same images', data.MINILM : 'MiniLM text', data.QWEN : 'Qwen3 text',
          data.GEMINI : 'Gemini text', data.CLAP : 'audio + raw features', 'raw' : 'audio + raw features'}
families = {}
for x in datasets:
    if both <= kinds[x]:
        families.setdefault(FAMILY[data.REGISTRY[x]['encoder']], []).append(x)

def family_table(s):
    cols = [(f, what) for f in families for what in ['acc', 'rank']]
    def get(m, c):
        ds = families[c[0]]
        if c[1] == 'acc':
            vals = [val(s, m, x) for x in ds]
            return np.nan if np.isnan(vals).any() else float(np.mean(vals))
        return RANK[s].loc[m, ds].mean() if m in RANK[s].index else np.nan
    head = ['<tr><th rowspan="2">method</th>' + ''.join(f'<th colspan="2" class="grp">{html.escape(f)} <span class="dim">({len(ds)})</span></th>' for f, ds in families.items()) + '</tr>',
            '<tr>' + ''.join('<th>accuracy</th><th class="rk">mean rank</th>' for _ in families) + '</tr>']
    return table(head, body(cols, get), 'summary')

# per-dataset detail, one table per shot count
def detail(s):
    cols = []
    for g in ['dev', 'other']:
        if groups[g]:
            cols += [('ds', x) for x in groups[g]] + [('mean', g), ('rank', g)]
    def get(m, c):
        return val(s, m, c[1]) if c[0] == 'ds' else group_mean(s, m, c[1])[0] if c[0] == 'mean' else group_rank(s, m, c[1])
    h1 = '<tr><th rowspan="2">method</th>' + ''.join(
        f'<th>{html.escape(c[1].replace("_dinov2", "").replace("_minilm", "").replace("_gemini", "").replace("_", " "))}</th>' if c[0] == 'ds' else f'<th class="gm">mean<br>{c[1]}</th>' if c[0] == 'mean' else f'<th class="rk">mean rank<br>{c[1]}</th>' for c in cols) + '</tr>'
    h2 = '<tr>' + ''.join(f'<th class="enc">{ENC[data.REGISTRY[c[1]]["encoder"]]}</th>' if c[0] == 'ds' else f'<th class="enc {"gm" if c[0] == "mean" else "rk"}"></th>' for c in cols) + '</tr>'
    return table([h1, h2], body(cols, get), 'detail')

settings_rows = ''.join(f'<tr><th>{html.escape(t.split(" · ")[0] + " · " + NAMES[m] if k == "ours" else NAMES[m])}</th>' + ''.join(f'<td class="l">{html.escape(setting(m, s))}</td>' for s in shots) + '</tr>'
                        for k, t, ms in SECTIONS if k != 'ceil' for m in ms if m in methods and len({chosen.get((m, s)) for s in shots} | {''}) > 1)
SETTINGS = table(['<tr><th>method</th>' + ''.join(f'<th class="l">{s}-shot</th>' for s in shots) + '</tr>'], [settings_rows], 'settings')

seeds = d.groupby('dataset').seed.nunique()
missing_base = [x for x in datasets if 'baselines' not in kinds[x]]

CSS = """
:root { --bg:#f9f9f7; --surface:#fcfcfb; --ink:#1a1a18; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --line2:#c3c2b7; --ours:#eaf2fc; --best:#1c5cab; --ceil:#898781; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --line2:#383835; --ours:#16243a; --best:#6da7ec; --ceil:#898781; color-scheme: dark; } }
:root[data-theme="dark"] { --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --line2:#383835; --ours:#16243a; --best:#6da7ec; --ceil:#898781; color-scheme: dark; }
* { box-sizing: border-box }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:76rem; margin:0 auto; padding-block:2rem 4rem; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-size:1.6rem; margin:0 0 .3rem; letter-spacing:-.01em } h2 { font-size:1.15rem; margin:2.4rem 0 .5rem } p, ul { max-width:46rem; color:var(--ink2); margin:.3rem 0 .8rem } li { margin-bottom:.3rem }
.wrap { overflow-x:auto; background:var(--surface); border:1px solid var(--line); border-radius:8px; }
table { border-collapse:collapse; font-size:.86rem; font-variant-numeric:tabular-nums; }
th, td { padding:.3rem .7rem; white-space:nowrap; border-bottom:1px solid var(--line); }
thead th { color:var(--ink2); font-weight:600; text-align:right; border-bottom:1px solid var(--line2); vertical-align:bottom; background:var(--surface); }
thead th:first-child, tbody th { text-align:left; position:sticky; left:0; background:var(--surface); font-weight:400; }
thead th.grp { text-align:center; border-left:1px solid var(--line2) } thead th.enc { font-weight:400; color:var(--muted); font-size:.75rem; } thead th.l, td.l { text-align:left }
td { text-align:right } td.best { font-weight:700; color:var(--best) } td.gm, th.gm { border-left:1px solid var(--line2); border-right:1px solid var(--line2); }
td.rk, th.rk { color:var(--ink2); border-right:1px solid var(--line2); } td.rk.best { color:var(--best) }
tr.sec th { font-weight:600; color:var(--ink2); font-size:.78rem; letter-spacing:.02em; padding-top:.7rem; background:var(--surface); }
tr.ours th, tr.ours td { background:var(--ours) } tr.ceil td, tr.ceil th { color:var(--ceil) }
.dim { color:var(--muted); font-weight:400 } code { font-family:ui-monospace,Menlo,monospace; font-size:.9em }
"""

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Few-shot Results</title><style>{CSS}</style></head><body><main>
<h1>Few-shot classification of frozen vectors</h1>
<p>Accuracy (%) on the unlabeled points, averaged over seeds. At most {d.n.max():,} points per dataset. Bold marks the best in each column, ceilings excluded. Our methods are shaded.</p>

<h2>Summary: baselines against ours</h2>
<p>Means over the datasets where both the baselines and our methods have been run. <strong>Mean rank</strong>: within each dataset the {len(real_methods)} methods are ranked by accuracy (1 is best), and the ranks are averaged over datasets. It is less swayed than the mean by one dataset where a method collapses.</p>
{SUMMARY}

{"".join(f'<h2>{s} labeled example{"s" if s > 1 else ""} per class, by where the vectors come from</h2><p>The same comparison, split by the encoder that produced the vectors. The number in brackets is how many datasets are in the group.</p>{family_table(s)}' for s in shots)}

{"".join(f'<h2>{s} labeled example{"s" if s > 1 else ""} per class, by dataset</h2><p>The second header row is the encoder that produced the vectors. Scroll sideways for more datasets.</p>{detail(s)}' for s in shots)}

<h2>Settings in use</h2>
<p>Only methods with a grid are listed. Each setting is the one with the best mean accuracy on the dev datasets for that shot count.</p>
{SETTINGS}

<h2>How to read this</h2>
<ul>
<li><strong>Two graphs.</strong> “Plain kNN graph”: each point is linked to its 10 most similar points by cosine similarity. “Under the cluster metric”: the same construction, after re-weighting the vector space with a metric learned from k-means clusters (no labels).</li>
<li><strong>Three predictors</strong>, usable on either graph. Spread the one-hot labels over the graph and take the largest. Or spread the vectors and train a linear SVM on the labeled rows. Or add the two methods’ standardized class scores.</li>
<li><strong>Dev datasets</strong> are the six CLIP image datasets every setting was chosen on, for baselines and ours alike, so those columns are tuned on the data they report. <strong>Other datasets</strong> never influence a setting.</li>
<li><strong>Seeds:</strong> {", ".join(f"{k} {v}" for k, v in seeds.reindex(datasets).items())}. Each seed redraws the sample and the labeled points; every method sees the same draws.</li>
{"<li><strong>Baselines not run yet on:</strong> " + ", ".join(missing_base) + ".</li>" if missing_base else ""}
</ul>
</main></body></html>
"""
out = f'{HERE}/results/summary.html'
open(out, 'w').write(page)

# --
# Short text summary for the terminal

w = max(len(NAMES[m]) for m in methods) + 2
print(f'{"accuracy % (mean rank)":{w + 2}s}' + '  '.join(f'{g + " " + str(s) + "-shot":>14s}' for g, s in gs))
for kind, title, ms in SECTIONS:
    ms = [m for m in ms if m in methods]
    if ms:
        print(title)
        for m in ms:
            cells = []
            for g, s in gs:
                a_, r_ = sum_get(m, (g, s, 'acc')), sum_get(m, (g, s, 'rank'))
                cells.append(f'{pct(a_):>6s} ' + (f'({r_:4.1f})' if not np.isnan(r_) else '      '))
            print(f'  {NAMES[m]:{w}s}' + '  '.join(f'{c:>14s}' for c in cells))
print(f'\n(datasets with both runs: dev {n_cmp["dev"]}, other {n_cmp["other"]}; rank is among {len(real_methods)} methods)\nwrote {out}')
