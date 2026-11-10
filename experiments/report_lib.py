"""
  report_lib.py - tiny inline-SVG / HTML helpers for make_report.py (no dependencies beyond numpy)

  Every chart is drawn against CSS custom properties, so light / dark themes swap in one place,
  and every mark carries a `data-tip` that the page's one tooltip script reads on hover and focus.
"""

import html
import numpy as np

esc = html.escape

def f3(v):
    return '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.3f}'

def sgn(v):
    return '–' if v is None or np.isnan(v) else f'{v:+.3f}'

def _ticks(lo, hi, n=5):
    span = hi - lo
    step = 10 ** np.floor(np.log10(span / n))
    for m in [1, 2, 2.5, 5, 10]:
        if span / (step * m) <= n:
            step = step * m
            break
    t0 = np.ceil(lo / step - 1e-9) * step
    return [round(float(t), 6) for t in np.arange(t0, hi + step * 1e-6, step)]

def _tickfmt(t, signed=False):
    if abs(t) < 1e-9:
        return '0'
    return f'{t:+.2f}' if signed else f'{t:.2f}'

# --
# Legend (HTML, so it wraps)

def legend(series):
    out = []
    for s in series:
        kind = s.get('kind', 'dot')
        if kind == 'dot':
            key = f'<span class="key key-dot" style="background:{s["color"]}"></span>'
        elif kind == 'tick':
            key = f'<span class="key key-tick" style="background:{s["color"]}"></span>'
        elif kind == 'ring':
            key = f'<span class="key key-ring" style="border-color:{s["color"]}"></span>'
        elif kind == 'line':
            key = f'<span class="key key-line" style="background:{s["color"]}"></span>'
        elif kind == 'bar':
            key = f'<span class="key key-bar" style="background:{s["color"]}"></span>'
        out.append(f'<span class="legend-item">{key}{esc(s["label"])}</span>')
    return f'<div class="legend">{"".join(out)}</div>'

# --
# Dot plot: one row per category, several marks per row on a shared x axis

def dotplot(rows, series, get, xlim, title='', signed=False, zero=None, w=470, label_w=104, row_h=30):
    """ rows: [(key, label)] ; series: [dict(key, label, color, kind)] ; get(row_key, series_key) -> float | None """
    top, bot, right = 30, 30, 16
    h   = top + row_h * len(rows) + bot
    x0, x1 = label_w, w - right
    sx  = lambda v: x0 + (min(max(v, xlim[0]), xlim[1]) - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
    o   = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">']
    if title:
        o.append(f'<text class="ptitle" x="{x0}" y="14">{esc(title)}</text>')

    for t in _ticks(*xlim):
        cls = 'axis' if (zero is not None and abs(t - zero) < 1e-9) else 'grid'
        o.append(f'<line class="{cls}" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 4}" y2="{h - bot + 4}"/>')
        o.append(f'<text class="tick" x="{sx(t):.1f}" y="{h - bot + 18}" text-anchor="middle">{_tickfmt(t, signed)}</text>')

    for i, (rk, rl) in enumerate(rows):
        y    = top + row_h * i + row_h / 2
        vals = [(s, get(rk, s['key'])) for s in series]
        vals = [(s, v) for s, v in vals if v is not None and not np.isnan(v)]
        o.append(f'<text class="rowlab" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{esc(rl)}</text>')
        if vals:
            lo, hi = min(v for _, v in vals), max(v for _, v in vals)
            o.append(f'<line class="span" x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')

        # ticks first, dots on top
        for s, v in sorted(vals, key=lambda sv: {'tick' : 0, 'ring' : 1}.get(sv[0].get('kind', 'dot'), 2)):
            tip  = f'{rl} · {s["label"]}: {sgn(v) if signed else f3(v)}'
            clip = v < xlim[0] or v > xlim[1]
            if clip:
                tip += ' (off the axis)'
            if s.get('kind', 'dot') == 'tick':
                o.append(f'<g class="mark" tabindex="0" data-tip="{esc(tip)}"><rect class="hit" x="{sx(v) - 8:.1f}" y="{y - 12:.1f}" width="16" height="24"/>'
                         f'<rect x="{sx(v) - 1.5:.1f}" y="{y - 9:.1f}" width="3" height="18" rx="1.5" fill="{s["color"]}"/></g>')
            elif s.get('kind') == 'ring':   # hollow, so a dot at the same value stays visible inside it
                o.append(f'<g class="mark" tabindex="0" data-tip="{esc(tip)}"><circle class="hit" cx="{sx(v):.1f}" cy="{y:.1f}" r="13"/>'
                         f'<circle class="ring" cx="{sx(v):.1f}" cy="{y:.1f}" r="8.5" stroke="{s["color"]}"/></g>')
            else:
                o.append(f'<g class="mark" tabindex="0" data-tip="{esc(tip)}"><circle class="hit" cx="{sx(v):.1f}" cy="{y:.1f}" r="12"/>'
                         f'<circle class="dot" cx="{sx(v):.1f}" cy="{y:.1f}" r="5.5" fill="{s["color"]}"/></g>')

    o.append('</svg>')
    return ''.join(o)

# --
# Small-multiple line panel: a few series over an ordered x, plus an optional horizontal reference

def linepanel(title, xs, series, get, ref=None, ref_label='', logx=False, xlabel='', w=250, h=178, xticks=None):
    """ series: [dict(key, label, color)] ; get(series_key, x) -> float | None ; ref: float """
    left, right, top, bot = 40, 14, 26, 34
    vals = [get(s['key'], x) for s in series for x in xs]
    vals = [v for v in vals if v is not None and not np.isnan(v)] + ([ref] if ref is not None else [])
    if not vals:
        return ''
    lo, hi = min(vals), max(vals)
    pad    = max((hi - lo) * 0.12, 0.004)
    lo, hi = lo - pad, hi + pad
    tx     = (lambda x: np.log10(x)) if logx else (lambda x: x)
    xa, xb = tx(min(xs)), tx(max(xs))
    sx     = lambda x: left + (tx(x) - xa) / (xb - xa) * (w - left - right)
    sy     = lambda v: h - bot - (v - lo) / (hi - lo) * (h - top - bot)

    o = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">',
         f'<text class="ptitle" x="{left}" y="14">{esc(title)}</text>']
    for t in _ticks(lo, hi, 4):
        o.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        o.append(f'<text class="tick" x="{left - 6}" y="{sy(t) + 3.5:.1f}" text-anchor="end">{t:.2f}</text>')

    for x in (xticks or xs):
        o.append(f'<text class="tick" x="{sx(x):.1f}" y="{h - bot + 15}" text-anchor="middle">{x:g}</text>')
    if xlabel:
        o.append(f'<text class="tick" x="{(left + w - right) / 2:.1f}" y="{h - 4}" text-anchor="middle">{esc(xlabel)}</text>')

    if ref is not None:
        o.append(f'<line class="ref" x1="{left}" x2="{w - right}" y1="{sy(ref):.1f}" y2="{sy(ref):.1f}"/>')
        o.append(f'<g class="mark" tabindex="0" data-tip="{esc(title)} · {esc(ref_label)}: {f3(ref)}"><rect class="hit" x="{left}" y="{sy(ref) - 6:.1f}" width="{w - left - right}" height="12"/></g>')

    for s in series:
        pts = [(x, get(s['key'], x)) for x in xs]
        pts = [(x, v) for x, v in pts if v is not None and not np.isnan(v)]
        if not pts:
            continue
        d = ' '.join(f'{"M" if i == 0 else "L"}{sx(x):.1f},{sy(v):.1f}' for i, (x, v) in enumerate(pts))
        o.append(f'<path class="line" d="{d}" stroke="{s["color"]}"/>')
        for x, v in pts:
            tip = f'{title} · {s["label"]} · {xlabel} {x:g}: {f3(v)}'
            o.append(f'<g class="mark" tabindex="0" data-tip="{esc(tip)}"><circle class="hit" cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="10"/>'
                     f'<circle class="dot" cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="4" fill="{s["color"]}"/></g>')

    o.append('</svg>')
    return ''.join(o)

# --
# Diverging bars: one bar per row from a zero baseline, optional per-dataset dots for spread

def divbars(rows, xlim, title='', w=760, label_w=350, row_h=30):
    """ rows: [dict(label, value, points=[(name, v)])] """
    top, bot, right = 14, 30, 44
    h   = top + row_h * len(rows) + bot
    x0, x1 = label_w, w - right
    sx  = lambda v: x0 + (min(max(v, xlim[0]), xlim[1]) - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
    o   = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">']
    for t in _ticks(*xlim, 6):
        cls = 'axis' if abs(t) < 1e-9 else 'grid'
        o.append(f'<line class="{cls}" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 4}" y2="{h - bot + 4}"/>')
        o.append(f'<text class="tick" x="{sx(t):.1f}" y="{h - bot + 18}" text-anchor="middle">{_tickfmt(t, True)}</text>')

    for i, r in enumerate(rows):
        y, v = top + row_h * i + row_h / 2, r['value']
        o.append(f'<text class="rowlab" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{esc(r["label"])}</text>')
        if v is None or np.isnan(v):
            continue
        a, b  = sorted([sx(0), sx(v)])
        bw    = max(b - a, 1)
        color = 'var(--pos)' if v >= 0 else 'var(--neg)'
        rr    = min(4, bw / 2)
        if v >= 0:   # square at the baseline, rounded at the data end
            d = f'M{a:.1f},{y - 7:.1f} h{bw - rr:.1f} a{rr},{rr} 0 0 1 {rr},{rr} v{14 - 2 * rr:.1f} a{rr},{rr} 0 0 1 -{rr},{rr} h-{bw - rr:.1f} z'
        else:
            d = f'M{b:.1f},{y - 7:.1f} h-{bw - rr:.1f} a{rr},{rr} 0 0 0 -{rr},{rr} v{14 - 2 * rr:.1f} a{rr},{rr} 0 0 0 {rr},{rr} h{bw - rr:.1f} z'
        o.append(f'<g class="mark" tabindex="0" data-tip="{esc(r["label"])} · mean over datasets: {sgn(v)}">'
                 f'<rect class="hit" x="{a - 4:.1f}" y="{y - 12:.1f}" width="{bw + 8:.1f}" height="24"/><path d="{d}" fill="{color}"/></g>')
        for name, pv in r.get('points', []):
            if pv is None or np.isnan(pv):
                continue
            o.append(f'<g class="mark" tabindex="0" data-tip="{esc(r["label"])} · {esc(name)}: {sgn(pv)}">'
                     f'<circle class="hit" cx="{sx(pv):.1f}" cy="{y:.1f}" r="8"/><circle class="pt" cx="{sx(pv):.1f}" cy="{y:.1f}" r="3"/></g>')
        lx = x1 + 6
        o.append(f'<text class="val" x="{lx:.1f}" y="{y + 4:.1f}">{sgn(v)}</text>')

    o.append('</svg>')
    return ''.join(o)

# --
# Scatter with edges (the toy figure)

def toy(points, labels, edges, marked, title, lim, w=300, h=260):
    pad = 16
    sx  = lambda v: pad + (v - lim[0]) / (lim[1] - lim[0]) * (w - 2 * pad)
    sy  = lambda v: h - pad - (v - lim[2]) / (lim[3] - lim[2]) * (h - 2 * pad - 14)
    col = ['var(--s1)', 'var(--s2)', 'var(--s3)']
    o   = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">',
           f'<text class="ptitle" x="{pad}" y="14">{esc(title)}</text>']
    for i, j in edges:
        cls = 'edge' if labels[i] == labels[j] else 'edge edge-x'
        o.append(f'<line class="{cls}" x1="{sx(points[i,0]):.1f}" y1="{sy(points[i,1]):.1f}" x2="{sx(points[j,0]):.1f}" y2="{sy(points[j,1]):.1f}"/>')
    for i, (p, l) in enumerate(zip(points, labels)):
        if i in marked:
            continue
        o.append(f'<circle class="dot-s" cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="3.2" fill="{col[l]}"/>')
    for i in marked:
        o.append(f'<circle class="lab-ring" cx="{sx(points[i,0]):.1f}" cy="{sy(points[i,1]):.1f}" r="8"/>'
                 f'<circle class="dot" cx="{sx(points[i,0]):.1f}" cy="{sy(points[i,1]):.1f}" r="4.5" fill="{col[labels[i]]}"/>')
    o.append('</svg>')
    return ''.join(o)

# --
# Tables

def table(header, rows, caption=None, best=None, num_from=1):
    """ rows: list of lists; best: set of (row_idx, col_idx) to embolden """
    best = best or set()
    o = ['<div class="tablewrap"><table>']
    if caption:
        o.append(f'<caption>{caption}</caption>')
    o.append('<thead><tr>' + ''.join(f'<th{" class=num" if i >= num_from else ""}>{h}</th>' for i, h in enumerate(header)) + '</tr></thead><tbody>')
    for i, r in enumerate(rows):
        cells = []
        for j, c in enumerate(r):
            txt = f3(c) if isinstance(c, (float, np.floating)) else str(c)
            if (i, j) in best:
                txt = f'<b>{txt}</b>'
            cells.append(f'<td{" class=num" if j >= num_from else ""}>{txt}</td>')
        o.append('<tr>' + ''.join(cells) + '</tr>')
    o.append('</tbody></table></div>')
    return ''.join(o)

def best_cells(rows, cols, skip_rows=()):
    """ (row, col) of the max in each numeric column, ignoring rows in skip_rows (e.g. oracles) """
    out = set()
    for j in cols:
        vals = [(r[j], i) for i, r in enumerate(rows) if i not in skip_rows and isinstance(r[j], (float, np.floating)) and not np.isnan(r[j])]
        if vals:
            m = max(v for v, _ in vals)
            out |= {(i, j) for v, i in vals if abs(v - m) < 5e-4}
    return out
