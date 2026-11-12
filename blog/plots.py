"""
  plots.py - sensitivity plots from blog/results/sens/*.csv

  usage (from repo root): python blog/plots.py        -> blog/results/sweeps.html

  One row of panels per setting of the cluster-metric graph. Every panel shows the change in accuracy, in points,
  relative to the plain kNN graph at its default settings (the zero line), averaged over the datasets of one encoder.
  One line per encoder, so it is easy to see which settings transfer and which are specific to CLIP.
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
FAM  = {data.CLIP : 'CLIP images', data.DINO : 'DINOv2 images', data.QWEN : 'Qwen3 text'}
COL  = {'CLIP images' : 'var(--s1)', 'DINOv2 images' : 'var(--s2)', 'Qwen3 text' : 'var(--s3)'}
PRED = {'ls' : 'spread the labels', 'fs' : 'spread the vectors + SVM'}

d = pd.concat([pd.read_csv(f) for f in glob.glob(f'{HERE}/results/sens/*.csv')], ignore_index=True)
d['fam'] = d.dataset.map(lambda x: FAM[data.REGISTRY[x]['encoder']])
fams     = [f for f in FAM.values() if f in set(d.fam)]

# reference: plain graph, default settings
ref = d[(d.factor == 'default') & (d.graph == 'plain')].groupby(['fam', 'dataset', 'pred', 'shots']).acc.mean()

def delta(factor, graph, value_as=float):
    """ mean over datasets of (accuracy - plain default), in points: index (fam, value, pred, shots) """
    x = d[(d.factor == factor) & (d.graph == graph)].copy()
    x['value'] = x['value'].map(value_as)
    m = x.groupby(['fam', 'dataset', 'pred', 'shots', 'value']).acc.mean()
    m = m - ref.reindex(m.droplevel('value').index).values
    return 100 * m.groupby(['fam', 'value', 'pred', 'shots']).mean()

# --
# SVG

def ticks(lo, hi, n=4):
    span = max(hi - lo, 1e-9)
    step = 10 ** np.floor(np.log10(span / n))
    for m in [1, 2, 5, 10]:
        if span / (step * m) <= n:
            step *= m
            break
    return [round(float(t), 6) for t in np.arange(np.ceil(lo / step) * step, hi + step * 1e-6, step)]

def panel(title, xs, get, logx=False, xlabel='', xticks=None, mark=None, w=250, h=176):
    """ get(fam, x) -> points. mark: the default x (drawn as a faint vertical line). """
    left, right, top, bot = 36, 12, 24, 34
    vals   = [get(f, x) for f in fams for x in xs]
    vals   = [v for v in vals if v is not None and not np.isnan(v)] + [0.0]
    lo, hi = min(vals), max(vals)
    pad    = max((hi - lo) * 0.12, 0.3)
    lo, hi = lo - pad, hi + pad
    tx     = (lambda x: np.log10(x)) if logx else (lambda x: x)
    xa, xb = tx(min(xs)), tx(max(xs))
    sx     = lambda x: left + (tx(x) - xa) / (xb - xa) * (w - left - right)
    sy     = lambda v: h - bot - (v - lo) / (hi - lo) * (h - top - bot)
    o = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">', f'<text class="ptitle" x="{left}" y="13">{html.escape(title)}</text>']
    for t in ticks(lo, hi):
        o.append(f'<line class="{"axis" if abs(t) < 1e-9 else "grid"}" x1="{left}" x2="{w - right}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        o.append(f'<text class="tick" x="{left - 5}" y="{sy(t) + 3.5:.1f}" text-anchor="end">{"0" if abs(t) < 1e-9 else f"{t:+g}"}</text>')
    if mark is not None:
        o.append(f'<line class="mark-x" x1="{sx(mark):.1f}" x2="{sx(mark):.1f}" y1="{top}" y2="{h - bot}"/>')
    for x in (xticks or xs):
        o.append(f'<text class="tick" x="{sx(x):.1f}" y="{h - bot + 14}" text-anchor="middle">{x:g}</text>')
    o.append(f'<text class="tick" x="{(left + w - right) / 2:.1f}" y="{h - 4}" text-anchor="middle">{html.escape(xlabel)}</text>')
    for f in fams:
        pts = [(x, get(f, x)) for x in xs]
        pts = [(x, v) for x, v in pts if v is not None and not np.isnan(v)]
        if not pts:
            continue
        o.append(f'<path class="line" stroke="{COL[f]}" d="' + ' '.join(f'{"M" if i == 0 else "L"}{sx(x):.1f},{sy(v):.1f}' for i, (x, v) in enumerate(pts)) + '"/>')
        for x, v in pts:
            tip = f'{f} · {title} · {xlabel} {x:g}: {v:+.1f} points vs plain graph'
            o.append(f'<g class="mark" tabindex="0" data-tip="{html.escape(tip)}"><circle class="hit" cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="9"/><circle class="dot" cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="3.6" fill="{COL[f]}"/></g>')
    o.append('</svg>')
    return ''.join(o)

LEGEND = '<div class="legend">' + ''.join(f'<span class="li"><span class="key" style="background:{COL[f]}"></span>{f}</span>' for f in fams) + \
         '<span class="li"><span class="key zero"></span>plain kNN graph, default settings</span><span class="li"><span class="key dflt"></span>default value</span></div>'

def row(title, note, factor, xlabel, graph='cmg', logx=False, xticks=None, mark=None, preds=('ls', 'fs')):
    m      = delta(factor, graph)
    xs     = sorted(m.index.get_level_values('value').unique())
    panels = []
    for shots in [1, 5]:
        for pred in preds:
            get = lambda f, x, pred=pred, shots=shots: m.get((f, x, pred, shots), np.nan)
            panels.append(panel(f'{shots}-shot, {PRED[pred]}', xs, get, logx=logx, xlabel=xlabel, xticks=xticks, mark=mark))
    return f'<h2>{html.escape(title)}</h2><p>{note}</p><figure>{LEGEND}<div class="panels">' + ''.join(f'<div>{p}</div>' for p in panels) + '</div></figure>'

def cat_table(title, note, factor, order=None):
    m    = delta(factor, 'cmg', value_as=str)
    vals = order or list(dict.fromkeys(d[(d.factor == factor) & (d.graph == 'cmg')]['value'].astype(str)))
    cols = [(f, s, p) for f in fams for s in [1, 5] for p in ['ls', 'fs']]
    head = '<tr><th rowspan="2">variant</th>' + ''.join(f'<th colspan="4" class="grp">{f}</th>' for f in fams) + '</tr><tr>' + \
           ''.join(f'<th>{s}-shot<br>{"labels" if p == "ls" else "vectors"}</th>' for f, s, p in cols) + '</tr>'
    body = ''
    for v in vals:
        cells = ''.join(f'<td class="{"pos" if (x := m.get((f, v, p, s), np.nan)) > 0.5 else "neg" if x < -0.5 else ""}">{"–" if np.isnan(x) else f"{x:+.1f}"}</td>' for f, s, p in cols)
        body += f'<tr><th>{html.escape(v)}</th>{cells}</tr>'
    return f'<h2>{html.escape(title)}</h2><p>{note}</p><div class="wrap"><table><thead>{head}</thead><tbody>{body}</tbody></table></div>'

CSS = """
:root { --bg:#f9f9f7; --surface:#fcfcfb; --ink:#1a1a18; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --pos:#1c5cab; --neg:#b3261e; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#52514e; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --pos:#6da7ec; --neg:#e66767; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#52514e; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --pos:#6da7ec; --neg:#e66767; color-scheme:dark; }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:72rem; margin:0 auto; padding-block:2rem 4rem; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-size:1.6rem; margin:0 0 .3rem } h2 { font-size:1.12rem; margin:2.2rem 0 .3rem } p { max-width:48rem; color:var(--ink2); margin:.2rem 0 .7rem }
figure { margin:0; background:var(--surface); border:1px solid var(--grid); border-radius:8px; padding:.9rem 1rem; }
.panels { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 14rem), 1fr)); gap:.6rem 1.2rem; } .panels > div { min-width:0 }
.legend { display:flex; flex-wrap:wrap; gap:.3rem 1.1rem; font-size:.82rem; color:var(--ink2); margin-bottom:.6rem } .li { display:inline-flex; align-items:center; gap:.4rem }
.key { width:16px; height:2px; border-radius:1px; display:inline-block } .key.zero { background:var(--axis); height:2px } .key.dflt { width:2px; height:13px; background:var(--grid) }
.chart { width:100%; height:auto; display:block; font-family:inherit } .chart text { fill:var(--ink2); font-size:10.5px } .chart .ptitle { fill:var(--ink); font-size:11.5px; font-weight:600 }
.chart .tick { fill:var(--muted); font-variant-numeric:tabular-nums } .chart .grid { stroke:var(--grid) } .chart .axis { stroke:var(--axis); stroke-width:1.5 } .chart .mark-x { stroke:var(--grid); stroke-width:2 }
.chart .line { fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round } .chart .dot { stroke:var(--surface); stroke-width:1.5 } .chart .hit { fill:transparent } .chart .mark { outline:none }
.chart .mark:hover .dot, .chart .mark:focus-visible .dot { stroke:var(--ink) }
.wrap { overflow-x:auto; background:var(--surface); border:1px solid var(--grid); border-radius:8px } table { border-collapse:collapse; font-size:.84rem; font-variant-numeric:tabular-nums }
th, td { padding:.3rem .65rem; border-bottom:1px solid var(--grid); white-space:nowrap; text-align:right } thead th { color:var(--ink2); font-weight:600; vertical-align:bottom } th.grp { text-align:center; border-left:1px solid var(--axis) }
tbody th, thead th:first-child { text-align:left; font-weight:400 } td.pos { color:var(--pos); font-weight:600 } td.neg { color:var(--neg) }
#tip { position:fixed; z-index:9; pointer-events:none; background:var(--ink); color:var(--bg); font-size:.78rem; padding:.3rem .5rem; border-radius:6px; max-width:20rem }
"""
JS = """
(function(){var t=document.getElementById('tip');function s(e,x,y){t.textContent=e.getAttribute('data-tip');t.hidden=false;t.style.left=Math.max(8,Math.min(innerWidth-t.offsetWidth-8,x+12))+'px';t.style.top=(y-t.offsetHeight-10<8?y+16:y-t.offsetHeight-10)+'px'}
document.addEventListener('pointermove',function(e){var m=e.target.closest&&e.target.closest('[data-tip]');if(m)s(m,e.clientX,e.clientY);else t.hidden=true});
document.addEventListener('focusin',function(e){var m=e.target.closest&&e.target.closest('[data-tip]');if(m){var r=m.getBoundingClientRect();s(m,r.left+r.width/2,r.top)}});
document.addEventListener('focusout',function(){t.hidden=true})})();
"""

n_runs = d.groupby('fam').apply(lambda g: f'{g.dataset.nunique()} datasets × {g.seed.nunique()} seeds', include_groups=False)
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Cluster-Metric Sweeps</title><style>{CSS}</style></head><body><main>
<h1>How the cluster-metric graph responds to its settings</h1>
<p>Each panel shows the change in accuracy, in points, from using the cluster-metric graph instead of the plain kNN graph. Zero is the plain graph at default settings;
above zero the metric helps. One line per encoder ({"; ".join(f"{f}: {n_runs[f]}" for f in fams)}). The faint vertical line marks the default value, which was tuned on CLIP.</p>
{row('Number of k-means clusters', 'Too few clusters hurts on every encoder. CLIP levels off by 100; DINOv2 is still improving at 800.', 'K', 'clusters K', logx=True, xticks=[10, 50, 200, 800], mark=100)}
{row('Metric strength λ', '0 is the plain graph, 1 uses the learned metric alone. CLIP wants a strong metric; DINOv2 and Qwen3 do best with a light one.', 'lam', 'λ', mark=0.5)}
{row('Ridge penalty (per point)', 'How strongly the regression onto cluster labels is regularized. The optimum differs by encoder, and for Qwen3 the default sits in a dip.', 'penalty', 'penalty', logx=True, xticks=[2e-6, 2e-4, 2e-2], mark=2e-4)}
{row('Neighbors k, cluster-metric graph', 'Relative to the plain graph at k = 10.', 'k', 'neighbors k', logx=True, xticks=[5, 10, 20, 50], mark=10)}
{row('Neighbors k, plain graph', 'The same sweep without the metric, for comparison.', 'k', 'neighbors k', graph='plain', logx=True, xticks=[5, 10, 20, 50], mark=10)}
{row('Restart probability α when spreading labels', 'Cluster-metric graph. Relative to the plain graph at α = 0.95.', 'alpha_ls', 'α', mark=0.95, preds=('ls',))}
{row('Restart probability α when spreading vectors', 'Cluster-metric graph. Relative to the plain graph at α = 0.85.', 'alpha_fs', 'α', mark=0.85, preds=('fs',))}
{row('Rounds', 'Re-cluster on vectors spread over the current graph and rebuild.', 'rounds', 'rounds', mark=1)}
{cat_table('What the metric is learned from', 'Clusters only (label-free), the few-shot classifier’s predictions for every point, or both. Points gained over the plain graph.', 'targets', ['clusters', 'pseudo', 'both'])}
{cat_table('What kind of metric', 'All built from the same 100 k-means clusters where clusters are used. “none” is the plain graph.', 'metric')}
{cat_table('Where the metric is fit', 'On the raw vectors (default), on vectors already spread over the plain graph, or no metric at all and the graph simply rebuilt on spread vectors.', 'fit_on')}
{cat_table('Clustering variance', 'Five different k-means seeds on the same data: how much the result moves with the clustering alone.', 'kmeans_seed')}
</main><div id="tip" hidden></div><script>{JS}</script></body></html>"""
out = f'{HERE}/results/sweeps.html'
open(out, 'w').write(page)
print(f'wrote {out}')
