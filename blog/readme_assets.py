"""
  readme_assets.py - generate the README's results tables and figures from blog/results/

  usage (from repo root): python blog/readme_assets.py

  Writes assets/ranks.svg and assets/metric_strength.svg, and replaces everything between the
  <!-- results:start --> and <!-- results:end --> markers in README.md. Nothing in that block is typed by hand.

  Settings: each method's small grid is resolved on the six CLIP "dev" datasets, per shot count, exactly as in table.py.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FAMILY = {data.CLIP : 'CLIP', data.DINO : 'DINOv2', data.QWEN : 'Qwen3', data.GEMINI : 'Gemini', data.MINILM : 'MiniLM', data.CLAP : 'audio, raw', 'raw' : 'audio, raw'}
FAM_NOTE = {'CLIP' : 'images', 'DINOv2' : 'images', 'Qwen3' : 'text', 'Gemini' : 'text', 'MiniLM' : 'text', 'audio, raw' : ''}
ROWS = [   # (method key, label, group)
    ('svm',               '`LinearSVC`',                                  'scikit-learn'),
    ('self_training',     '`SelfTrainingClassifier`',                     'scikit-learn'),
    ('label_spreading',   '`LabelSpreading`',                             'scikit-learn'),
    ('kmeans_seeded',     '`KMeans`, seeded with the class means',        'scikit-learn'),
    ('randomwalk',        'random walk',                                  'graphlearning'),
    ('poisson_connected', 'Poisson learning (graph made connected)',      'graphlearning'),
    ('fs+ls',             '`metric=None`',                                'this library, `predictor="both"`'),
    ('cmg_fs+ls',         '`metric="cluster"`',                           'this library, `predictor="both"`'),
    ('plg_fs+ls',         '`metric="cluster+pseudo"` (the default)',      'this library, `predictor="both"`'),
    ('fs',                '`metric=None`',                                'this library, `predictor="vectors"`'),
    ('cmg_fs',            '`metric="cluster"`',                           'this library, `predictor="vectors"`'),
    ('plg_fs',            '`metric="cluster+pseudo"`',                    'this library, `predictor="vectors"`'),
    ('ls',                '`metric=None`',                                'this library, `predictor="labels"`'),
    ('cmg_ls',            '`metric="cluster"`',                           'this library, `predictor="labels"`'),
    ('plg_ls',            '`metric="cluster+pseudo"`',                    'this library, `predictor="labels"`'),
]
OURS = {'fs', 'cmg_fs', 'plg_fs', 'ls', 'cmg_ls', 'plg_ls', 'fs+ls', 'cmg_fs+ls', 'plg_fs+ls'}

# Our methods are reported at the LIBRARY DEFAULTS (alpha=0.85 for vectors, alpha_labels=0.95), the same at every shot count.
# Baselines get the best setting from their grid, chosen per shot count on the dev datasets.
def is_default(variant):
    v = variant.split(',', 1)[1] if ',' in variant else ''
    return v in ('alpha=0.85', 'alpha=0.95', 'alpha_ls=0.95,alpha_fs=0.85')
DEFAULT_ALPHA = {'fs' : 'alpha=0.85', 'cmg_fs' : 'alpha=0.85', 'plg_fs' : 'alpha=0.85', 'ls' : 'alpha=0.95', 'cmg_ls' : 'alpha=0.95', 'plg_ls' : 'alpha=0.95'}

# --
# Load + resolve settings on dev (same rules as table.py)

d = pd.concat([pd.read_csv(f) for f in glob.glob(f'{HERE}/results/baselines*/*.csv') + glob.glob(f'{HERE}/results/ours*/*.csv')], ignore_index=True)
d = d[d.dataset.isin(data.REGISTRY)].copy()
d['variant'] = d.variant.fillna('')
d['group']   = d.dataset.map(lambda x: data.REGISTRY[x]['group'])
d['family']  = d.dataset.map(lambda x: FAMILY[data.REGISTRY[x]['encoder']])
ok = d.groupby(['dataset', 'kind']).seed.agg(set).groupby('dataset').agg(lambda s: set.intersection(*s))
d  = d[[seed in ok[ds] for ds, seed in zip(d.dataset, d.seed)]]

dev    = d[d.group == 'dev']
score  = dev.groupby(['method', 'shots', 'variant', 'dataset']).acc.mean().groupby(['method', 'shots', 'variant']).mean()
chosen = {k : v[2] for k, v in score.groupby(level=['method', 'shots']).idxmax().items()}
def kept(m, s, v):
    if m in OURS:
        tail = v.split(',', 1)[1] if ',' in v else ''
        return tail == DEFAULT_ALPHA.get(m, 'alpha_ls=0.95,alpha_fs=0.85')
    return chosen.get((m, s)) == v
keep   = d[[kept(m, s, v) for m, s, v in zip(d.method, d.shots, d.variant)]]
keep   = keep[~keep.method.str.startswith('oracle')]

families = [f for f in ['CLIP', 'DINOv2', 'Qwen3', 'Gemini', 'MiniLM', 'audio, raw'] if f in set(keep.family)]
n_fam    = keep.groupby('family').dataset.nunique()
datasets = sorted(keep.dataset.unique())
n_seeds  = keep.groupby('dataset').seed.nunique()

def acc_table(shots):
    k    = keep[keep.shots == shots]
    cell = k.groupby(['method', 'dataset']).acc.mean().unstack('dataset')
    rank = cell.rank(ascending=False, method='average')
    fam  = {ds : FAMILY[data.REGISTRY[ds]['encoder']] for ds in cell.columns}
    fam_acc  = 100 * cell.T.groupby(fam).mean().T
    return cell, rank, fam_acc

def md_table(shots):
    cell, rank, fam_acc = acc_table(shots)
    keys  = [m for m, _, _ in ROWS if m in cell.index]
    best  = fam_acc.loc[keys].max()
    brank = rank.loc[keys].mean(1).min()
    lines = ['| | ' + ' | '.join(f'{f}' for f in families) + ' | mean rank |', '|---|' + '---:|' * (len(families) + 1)]
    group = None
    for m, label, g in ROWS:
        if m not in cell.index:
            continue
        if g != group:
            lines.append(f'| **{g}** |' + ' |' * (len(families) + 1))
            group = g
        vals = [f'**{fam_acc.loc[m, f]:.1f}**' if abs(fam_acc.loc[m, f] - best[f]) < 0.05 else f'{fam_acc.loc[m, f]:.1f}' for f in families]
        r    = rank.loc[m].mean()
        lines.append(f'| {label} | ' + ' | '.join(vals) + (f' | **{r:.1f}** |' if abs(r - brank) < 0.05 else f' | {r:.1f} |'))
    return '\n'.join(lines)

# --
# Figure 1: rank of each method on every dataset (how often is a method near the top?)

W, INK, MUTED, GRID, BASE, OUR = '#ffffff', '#1a1a18', '#6b6a66', '#e4e3dd', '#8a8985', '#2a78d6'

def ranks_svg():
    rows   = [(m, lab.replace('`', ''), g) for m, lab, g in ROWS]
    panel_w, label_w, row_h, top, bot = 300, 338, 20, 46, 52
    n_meth = keep.method.nunique()
    H  = top + row_h * len(rows) + bot
    Wd = label_w + 2 * panel_w + 40
    o  = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {H}" width="{Wd}" font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
          f'<rect width="{Wd}" height="{H}" fill="{W}"/>']
    for p, shots in enumerate([1, 5]):
        _, rank, _ = acc_table(shots)
        x0 = label_w + p * (panel_w + 24)
        sx = lambda r: x0 + (r - 1) / (n_meth - 1) * panel_w
        o.append(f'<text x="{x0}" y="18" font-size="13" font-weight="600" fill="{INK}">{shots} labeled example{"s" if shots > 1 else ""} per class</text>')
        o.append(f'<text x="{x0}" y="34" font-size="11" fill="{MUTED}">rank among {n_meth} methods on each of {rank.shape[1]} datasets (1 = best)</text>')
        for t in [1, 5, 10, 15, 20]:
            if t <= n_meth:
                o.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 4}" y2="{H - bot + 4}" stroke="{GRID}"/>')
                o.append(f'<text x="{sx(t):.1f}" y="{H - bot + 16}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{t}</text>')
        for i, (m, lab, g) in enumerate(rows):
            if m not in rank.index:
                continue
            y   = top + row_h * i + row_h / 2
            col = OUR if m in OURS else BASE
            r   = rank.loc[m].dropna().values
            if p == 0:
                kind = '' if m not in OURS else '  · ' + ('both' if '+' in m else 'vectors' if 'fs' in m else 'labels')
                o.append(f'<text x="{label_w - 14}" y="{y + 4:.1f}" font-size="11.5" fill="{INK}" text-anchor="end">{lab}{kind}</text>')
            for v in r:
                o.append(f'<circle cx="{sx(v):.1f}" cy="{y:.1f}" r="3" fill="{col}" fill-opacity="0.28"/>')
            o.append(f'<rect x="{sx(r.mean()) - 1.5:.1f}" y="{y - 8:.1f}" width="3" height="16" rx="1.5" fill="{col}"/>')
    o.append(f'<circle cx="{label_w}" cy="{H - 9}" r="3" fill="{MUTED}" fill-opacity="0.4"/><text x="{label_w + 8}" y="{H - 5}" font-size="10.5" fill="{MUTED}">one dataset</text>')
    o.append(f'<rect x="{label_w + 86}" y="{H - 16}" width="3" height="14" rx="1.5" fill="{MUTED}"/><text x="{label_w + 95}" y="{H - 5}" font-size="10.5" fill="{MUTED}">mean rank</text>')
    o.append(f'<rect x="{label_w + 170}" y="{H - 14}" width="10" height="10" rx="2" fill="{OUR}"/><text x="{label_w + 185}" y="{H - 5}" font-size="10.5" fill="{MUTED}">this library</text>')
    o.append(f'<rect x="{label_w + 262}" y="{H - 14}" width="10" height="10" rx="2" fill="{BASE}"/><text x="{label_w + 277}" y="{H - 5}" font-size="10.5" fill="{MUTED}">scikit-learn / graphlearning</text>')
    o.append('</svg>')
    return '\n'.join(o)

# --
# Figure 2: the label-free cluster metric is tuned to CLIP (metric strength sweep)

def strength_svg():
    s = pd.concat([pd.read_csv(f) for f in glob.glob(f'{HERE}/results/sens/*.csv')], ignore_index=True)
    s['family'] = s.dataset.map(lambda x: FAMILY[data.REGISTRY[x]['encoder']])
    ref = s[(s.factor == 'default') & (s.graph == 'plain')].groupby(['family', 'dataset', 'pred', 'shots']).acc.mean()
    x   = s[(s.factor == 'lam') & (s.graph == 'cmg')].copy()
    x['value'] = x['value'].astype(float)
    m   = x.groupby(['family', 'dataset', 'pred', 'shots', 'value']).acc.mean()
    m   = 100 * (m - ref.reindex(m.droplevel('value').index).values).groupby(['family', 'value', 'pred', 'shots']).mean()
    fams = [f for f in ['CLIP', 'DINOv2', 'Qwen3'] if f in set(x.family)]
    col  = {'CLIP' : '#2a78d6', 'DINOv2' : '#eb6834', 'Qwen3' : '#1baf7a'}
    xs   = sorted(x.value.unique())
    pw, ph, left, top, gap = 250, 150, 44, 44, 30
    Wd, H = left + 2 * pw + gap + 76, top + ph + 60
    lo, hi = float(m.min()) - 0.6, float(m.max()) + 0.6
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {H}" width="{Wd}" font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">', f'<rect width="{Wd}" height="{H}" fill="{W}"/>']
    for p, shots in enumerate([1, 5]):
        x0 = left + p * (pw + gap)
        sx = lambda v: x0 + v * pw
        sy = lambda v: top + ph - (v - lo) / (hi - lo) * ph
        o.append(f'<text x="{x0}" y="18" font-size="13" font-weight="600" fill="{INK}">{shots} labeled example{"s" if shots > 1 else ""} per class</text>')
        o.append(f'<text x="{x0}" y="34" font-size="11" fill="{MUTED}">accuracy points gained over the plain graph</text>')
        for t in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
            if t % 2 == 0:
                o.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{"#b9b8b0" if t == 0 else GRID}" stroke-width="{1.5 if t == 0 else 1}"/>')
                if p == 0:
                    o.append(f'<text x="{x0 - 6}" y="{sy(t) + 3.5:.1f}" font-size="10.5" fill="{MUTED}" text-anchor="end">{"0" if t == 0 else f"{t:+d}"}</text>')
        for v in xs:
            o.append(f'<text x="{sx(v):.1f}" y="{top + ph + 15}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{v:g}</text>')
        o.append(f'<text x="{x0 + pw / 2:.1f}" y="{top + ph + 31}" font-size="10.5" fill="{MUTED}" text-anchor="middle">metric_strength  (0 = plain graph, 0.5 = default)</text>')
        ends = []
        for f in fams:
            pts = [(v, m.get((f, v, 'fs', shots), np.nan)) for v in xs]
            o.append(f'<path fill="none" stroke="{col[f]}" stroke-width="2" stroke-linejoin="round" d="' + ' '.join(f'{"M" if i == 0 else "L"}{sx(v):.1f},{sy(a):.1f}' for i, (v, a) in enumerate(pts)) + '"/>')
            for v, a in pts:
                o.append(f'<circle cx="{sx(v):.1f}" cy="{sy(a):.1f}" r="3.4" fill="{col[f]}" stroke="{W}" stroke-width="1.5"/>')
            if p == 1:
                ends.append((sy(pts[-1][1]), f))
        ends.sort()
        for i in range(1, len(ends)):                      # push overlapping labels apart
            if ends[i][0] - ends[i - 1][0] < 14:
                ends[i] = (ends[i - 1][0] + 14, ends[i][1])
        for yy, f in ends:
            o.append(f'<circle cx="{x0 + pw + 10}" cy="{yy - 4:.1f}" r="4" fill="{col[f]}"/><text x="{x0 + pw + 18}" y="{yy:.1f}" font-size="11" fill="{INK}">{f}</text>')
    o.append(f'<text x="{left}" y="{H - 6}" font-size="10.5" fill="{MUTED}">predictor="vectors". Mean over six datasets per encoder, 3 seeds.</text></svg>')
    return '\n'.join(o)

# --
# Write

os.makedirs(f'{ROOT}/assets', exist_ok=True)
open(f'{ROOT}/assets/ranks.svg', 'w').write(ranks_svg())
open(f'{ROOT}/assets/metric_strength.svg', 'w').write(strength_svg())

def robustness():
    lines = []
    for shots in [1, 5]:
        cell, rank, _ = acc_table(shots)
        base  = [m for m in cell.index if m not in OURS]
        top5  = (rank <= 5).sum(1)
        bb    = top5[base].idxmax()
        gap   = 100 * (cell.loc['plg_fs+ls'] - cell.loc[base].max())                 # vs the best baseline, picked per dataset in hindsight
        bad   = gap[gap < -5].sort_values()
        label = dict((m, l.replace('`', '')) for m, l, _ in ROWS)
        lines.append(f'- **{shots}-shot:** the default is among the top 5 of {len(cell)} methods on {int(top5["plg_fs+ls"])} of {cell.shape[1]} datasets; '
                     f'the most consistent baseline ({label.get(bb, bb)}) manages {int(top5[bb])}. '
                     f'Against the best baseline chosen separately for every dataset with hindsight, the default is {gap.mean():+.1f} points on average and ahead on {int((gap > 0).sum())} of {cell.shape[1]}. '
                     + (f'It is more than 5 points behind on: ' + ', '.join(f'{k} ({v:+.1f}, behind {label.get(cell.loc[base, k].idxmax(), cell.loc[base, k].idxmax())})' for k, v in bad.items()) + '.' if len(bad) else 'It is never more than 5 points behind.'))
    return '\n'.join(lines)

fam_line = ', '.join(f'{f} ({FAM_NOTE[f] + ", " if FAM_NOTE[f] else ""}{n_fam[f]})' for f in families)
block = f"""<!-- results:start -->
<!-- generated by blog/readme_assets.py from blog/results/ : do not edit by hand -->

Accuracy (%) on the unlabeled points, averaged over the datasets in each group and over {int(n_seeds.min())}{"" if n_seeds.min() == n_seeds.max() else "–" + str(int(n_seeds.max()))} seeds.
Groups are named by what produced the vectors: {fam_line}. **Mean rank** is over all {len(datasets)} dataset / encoder combinations,
among the {keep.method.nunique()} methods compared (1 is best). Bold marks the best entry in a column.
Rows for this library use its default settings throughout; each baseline gets the best setting from a small grid, chosen per shot count on the CLIP datasets.

**1 labeled example per class**

{md_table(1)}

**5 labeled examples per class**

{md_table(5)}

![Rank of each method on every dataset](assets/ranks.svg)

{robustness()}

<!-- results:end -->"""

readme = f'{ROOT}/README.md'
text   = open(readme).read()
if '<!-- results:start -->' in text:
    a, b = text.index('<!-- results:start -->'), text.index('<!-- results:end -->') + len('<!-- results:end -->')
    open(readme, 'w').write(text[:a] + block + text[b:])
    print('updated the results block in README.md')
else:
    print('README.md has no results markers; block follows\n')
print(block)
