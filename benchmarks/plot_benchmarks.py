"""
plot_benchmarks.py -- comparison figure and summary table from results/benchmark_results.csv (and
the S-versus-time records of the SOR runs in results/sor_trajectories.csv).

    python plot_benchmarks.py                 (needs matplotlib for the figure; the table needs only numpy)
    python plot_benchmarks.py --csv some_other_results.csv

* Errors are recomputed from S with the CURRENT reference in bench_common.py (S_REF), so rows logged
  before the reference was updated are still compared correctly.
* When the same method and rung were run more than once, the latest row counts.
* Rows of the reference calculation and of the lower bounds are not plotted (they are not methods
  one would use to get S; they bracket the true value).

Figure (results/benchmark_comparison.png and .pdf), two panels with the same error axis (and a third):
  left   time to the FIRST S: setup (grid/mesh, matrices, Fourier tables) + one solve.
  right  time for each FURTHER S with new dielectric constants, setup reused (the cost that
         matters when a whole spectrum is analyzed); only for methods where that is meaningful
         (SOR has no reusable setup and is left out).
  error  S/S_ref - 1, log scale.  All methods except the bound average give upper bounds (error > 0);
         a negative error (bound average only) is plotted as its absolute value with a hollow marker.
  The dashed line is the model's error; thin lines in the left panel are SOR runs as they iterate
  (they start far above the plotted range, which is cut off at an error of 1%).

  third  (only when the CSV has convergence-check rows) for each method, its cheapest rung at the
         model's accuracy: first S (filled marker) and first S + one convergence check (open marker;
         the same method once more with every element size halved, bench_common.record_check), with
         the ratio; the dashed line is the model's first S without a check.

Tables (printed and saved to results/summary.txt):
  1. for each method, the fastest run that is at least as accurate as the model, its times, the ratio
     to the model's times, and 'first S + check' with its ratio to the model's first S + check;
  2. the convergence checks: rung and check nodes, the check's estimate of the rung's error
     (S_rung/S_check - 1) against the true error, and what the check adds to the first S.
Rows labelled 'check: ...' are never chosen as a method's run; CSV files without them (older runs)
give the same table and figure as before, with '-' in the check columns.
"""
import bench_common as bc          # first: one thread for every numerical library

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np

# Plot order = categorical colour order (validated palette, light mode) and marker shape, so the
# methods stay distinguishable without colour (shapes) and in print.
METHODS = OrderedDict([
    ('model (ide_model.py)',              ('model (ide_model.py)',               '#2a78d6', 'o')),
    ('fem2d.py (bilinear, direct)',       ('FEM, bilinear (fem2d.py)',           '#eb6834', 's')),
    ('FEM biquadratic (upper bound)',     ('FEM, biquadratic',                   '#1baf7a', 'D')),
    ('FEM bound average (bilinear)',      ('FEM, average of bilinear bounds',    '#eda100', '^')),
    ('five-point, direct (graded grid)',  ('5-point, graded grid, direct',       '#e87ba4', 'v')),
    ('five-point, SOR (graded grid)',     ('5-point, graded grid, SOR',          '#008300', 'P')),
    ('SOR, uniform squares',              ('SOR, uniform squares',               '#6250d6', 'X')),
    ('SOR, uniform rectangles',           ('SOR, uniform 2:1 rectangles',        '#e34948', 'h')),
])
TUNED = 'fem2d.py, tuned mesh'           # one point, drawn in the fem2d colour with a star
MODEL = 'model (ide_model.py)'
SOR_METHODS = ('five-point, SOR (graded grid)', 'SOR, uniform squares', 'SOR, uniform rectangles')


def load(path):
    """Latest row per (method, label), with the error recomputed from S.  Returns (runs, checks): the
    rows of the methods' rungs, and the convergence-check rows (label 'check: <rung label>') keyed by
    (method, label of the rung they check)."""
    rows = OrderedDict()
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            m = r['method']
            if m not in METHODS and m != TUNED:
                continue
            S = float(r['S'])
            rows[(m, r['label'])] = dict(method=m, label=r['label'], n=int(float(r['n_nodes'])), S=S,
                                         err=bc.rel_err(S), first=float(r['total_s']),
                                         each=float(r['per_eval_s']) if r['per_eval_s'] else None,
                                         cpu=r.get('cpu', ''))
    runs = [r for r in rows.values() if not bc.is_check(r['label'])]
    checks = {(r['method'], r['label'][len(bc.CHECK_PREFIX):]): r for r in rows.values() if bc.is_check(r['label'])}
    return runs, checks


def as_accurate_as_model(r):
    err_model = bc.rel_err(bc.S_MODEL)
    if r['method'] == 'FEM bound average (bilinear)':      # not a bound: compare the size of the error
        return abs(r['err']) <= err_model
    return bc.as_accurate_as_model(r['S'])                   # upper bounds: S <= S_model


# The model's own row is its default rung, which defines the accuracy target, whatever the timings:
# 'default (exact sums)' (ide_model.py since 21 Nov 2026), or 'default (paper settings)' in older CSVs.
# (Its 'legacy' rung, sums cut off at 20,000 terms, is slightly lower than S_MODEL, so it would
# otherwise count as 'at least as accurate', and its timings are close to the default's.)
MODEL_DEFAULT_LABELS = ('default (exact sums)', 'default (paper settings)')


def chosen_run(rows, m):
    """The fastest (first S) run of method m that is at least as accurate as the model, or None.
    For the model itself: its default rung (MODEL_DEFAULT_LABELS)."""
    if m == MODEL:
        for label in MODEL_DEFAULT_LABELS:
            hit = [r for r in rows if r['method'] == m and r['label'] == label]
            if hit:
                return hit[0]
    ok = [r for r in rows if r['method'] == m and as_accurate_as_model(r)]
    return min(ok, key=lambda r: r['first']) if ok else None


def check_of(rows, checks, m, chosen):
    """(rung, check row, same) for method m: the logged check of the chosen rung (same = True); else the
    check of the method's fewest-nodes rung at the model's accuracy, which is what the benchmark scripts
    check (same = False); (None, None, False) if there is neither."""
    if chosen is not None and (m, chosen['label']) in checks:
        return chosen, checks[(m, chosen['label'])], True
    ok = [r for r in rows if r['method'] == m and as_accurate_as_model(r)]
    if ok:
        base = min(ok, key=lambda r: r['n'])
        if (m, base['label']) in checks:
            return base, checks[(m, base['label'])], False
    return None, None, False


def summary(rows, checks):
    ref = chosen_run(rows, MODEL)
    ref_base, ref_chk, _ = check_of(rows, checks, MODEL, ref)
    ref_fc = ref_base['first'] + ref_chk['first'] if ref_chk else None
    lines = [f"Fastest run of each method that is at least as accurate as the model "
             f"(model error {100 * bc.rel_err(bc.S_MODEL):+.4f}%; reference S = {bc.S_REF:.9f})", '',
             f"{'method':34s} {'rung':44s} {'nodes':>11s} {'error':>10s} {'first S':>10s} {'x model':>9s} "
             f"{'each S':>10s} {'x model':>9s} {'first+chk':>10s} {'x model':>9s}"]
    chk_lines, dagger = [], False
    for m in list(METHODS) + [TUNED]:
        r = chosen_run(rows, m)
        if r is None:
            tried = [r for r in rows if r['method'] == m]
            if tried:
                best = min(tried, key=lambda r: abs(r['err']))
                note = '(not reached; best error %+.4f%%)' % (100 * best['err'])
                lines.append(f'{m:34s} {note:44s}')
            continue
        x1 = f"{r['first'] / ref['first']:9.3g}" if ref else f"{'':>9s}"
        each = f"{r['each']:10.3g}" if r['each'] is not None else f"{'-':>10s}"
        x2 = (f"{r['each'] / ref['each']:9.3g}" if (ref and r['each'] is not None) else f"{'':>9s}")
        base, chk, same = check_of(rows, checks, m, r)
        if chk is None:
            fc, x3 = f"{'-':>10s}", f"{'':>9s}"
        else:
            t_fc = base['first'] + chk['first']
            fc = f"{t_fc:9.3g}" + (' ' if same else '+')
            dagger = dagger or not same
            x3 = f"{t_fc / ref_fc:9.3g}" if ref_fc else f"{'':>9s}"
            est = base['S'] / chk['S'] - 1.0
            chk_lines.append(f"{m + (' *' if m == MODEL else ''):34s} {base['n']:11,} {chk['n']:11,} {est:+11.2e} "
                             f"{base['err']:+11.2e} {base['first']:10.3g} {t_fc:10.3g} {t_fc / base['first']:7.3g} "
                             + (f"{t_fc / ref['first']:12.3g}" if ref else ''))
        lines.append(f"{m:34s} {r['label'][:44]:44s} {r['n']:11,} {100 * r['err']:+9.4f}% {r['first']:10.3g} {x1} "
                     f"{each} {x2} {fc} {x3}")
    lines.append('')
    lines.append("Times in seconds.  'first S' = setup + one solve; 'each S' = each further S with the setup reused.")
    lines.append("'first+chk' = first S + one convergence check: the same method once more with every element size")
    lines.append("halved, the least a user without a reference value must do to learn the rung's error.  Its 'x model'")
    lines.append("compares with the model's first S + its own check (in-plane node spacing halved), which sees only the")
    lines.append("model's discretization error (about 1e-5), not its modelling error (about 1.1e-4), known only from")
    lines.append("comparisons such as these.  The uniform grids (SOR squares and rectangles) have no check: their next")
    lines.append("grid would have about 93 million nodes.")
    if dagger:
        lines.append("+ : the check was made on the method's fewest-nodes rung at the model's accuracy (table 2), not on")
        lines.append("    the rung shown; first+chk is that rung's first S plus the check.")
    if chk_lines:
        lines += ['', 'Convergence checks (each method\'s fewest-nodes rung at the model\'s accuracy, then every element '
                  'size halved)', '',
                  f"{'method':34s} {'rung nodes':>11s} {'chk nodes':>11s} {'estimate':>11s} {'true error':>11s} "
                  f"{'first S':>10s} {'first+chk':>10s} {'ratio':>7s} {'/ model 1st':>12s}"]
        lines += chk_lines
        lines += ['',
                  "estimate = S_rung/S_check - 1, what the check tells a user about the rung's error; true error from",
                  "S_REF.  ratio = (first+chk)/(first S).  '/ model 1st' = first+chk divided by the model's first S",
                  "WITHOUT a check (the comparison for geometries inside the range where the model's error is known).",
                  "* the model's check sees only its discretization error, not its modelling error."]
    return '\n'.join(lines)


def trajectories(path):
    """SOR runs: (method, label) -> arrays of time and error."""
    out = OrderedDict()
    if not os.path.isfile(path):
        return out
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            t, S = float(r['time_s']), float(r['S'])
            if t <= 0:
                continue
            key = (r['method'], r['label'])
            if bc.is_check(r['label']):                      # convergence checks are not rungs
                continue
            if key not in out or t < out[key][0][-1]:        # a later run of the same rung replaces the earlier one
                out[key] = ([], [])
            out[key][0].append(t)
            out[key][1].append(bc.rel_err(S))
    return out


def check_pairs(rows, checks):
    """[(method, display name, colour, marker, rung, check row)] for the methods with a logged check, in
    plot order (the tuned fem2d mesh last)."""
    out = []
    for m in list(METHODS) + [TUNED]:
        base, chk, _ = check_of(rows, checks, m, chosen_run(rows, m))
        if chk is None:
            continue
        if m == TUNED:
            name, color, marker = 'FEM, bilinear, tuned mesh', METHODS['fem2d.py (bilinear, direct)'][1], '*'
        else:
            name, color, marker = METHODS[m]
        out.append((m, name, color, marker, base, chk))
    return out


def figure(rows, checks, traj, path_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid = '#2b2b2b', '#6b6b66', '#e4e3df'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': muted, 'axes.labelcolor': ink,
                         'xtick.color': muted, 'ytick.color': muted, 'axes.linewidth': 0.8})
    pairs = check_pairs(rows, checks)
    if pairs:       # a third panel for the convergence checks
        fig = plt.figure(figsize=(15.0, 4.8), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.85])
        a0 = fig.add_subplot(gs[0])
        axes = [a0, fig.add_subplot(gs[1], sharey=a0), fig.add_subplot(gs[2])]
        plt.setp(axes[1].get_yticklabels(), visible=False)
    else:           # older results without checks: the original two panels
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, constrained_layout=True)
    err_model = bc.rel_err(bc.S_MODEL)
    for ax, key, title in [(axes[0], 'first', 'Time to the first S (setup + one solve)'),
                           (axes[1], 'each', 'Time for each further S (setup reused)')]:
        if key == 'first':
            for (m, _), (t, e) in traj.items():
                if m in METHODS:
                    e = np.abs(np.array(e))
                    keep = e > 0
                    ax.plot(np.array(t)[keep], e[keep], color=METHODS[m][1], lw=0.7, alpha=0.35, zorder=1)
        for m, (name, color, marker) in METHODS.items():
            pts = sorted([r for r in rows if r['method'] == m and r[key] is not None and r['err'] != 0
                          and not r['label'].startswith('legacy')],      # the model's 20,000-term rung: CSV only
                         key=lambda r: r[key])
            if not pts:
                continue
            t = np.array([r[key] for r in pts])
            e = np.array([r['err'] for r in pts])
            ax.plot(t, np.abs(e), color=color, lw=1.5, zorder=2, label=name)
            ax.scatter(t[e > 0], e[e > 0], color=color, marker=marker, s=30, zorder=3,
                       edgecolors='white', linewidths=0.8)
            ax.scatter(t[e < 0], -e[e < 0], facecolors='white', edgecolors=color, marker=marker, s=30,
                       zorder=3, linewidths=1.2)
        tuned = [r for r in rows if r['method'] == TUNED and r[key] is not None]
        for k, r in enumerate(tuned):              # one legend entry, however many tuned meshes are logged
            ax.scatter([r[key]], [abs(r['err'])], color=METHODS['fem2d.py (bilinear, direct)'][1], marker='*',
                       s=110, zorder=4, edgecolors='white', linewidths=0.8,
                       label='FEM, bilinear, tuned mesh' if ax is axes[0] and k == 0 else None)
        ax.axhline(err_model, color=muted, lw=1.0, ls='--', zorder=0)
        ax.text(0.99, err_model / 1.15, "model's error", color=muted, ha='right', va='top',
                transform=ax.get_yaxis_transform(), fontsize=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='major', color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10, color=ink, loc='left')
        ax.set_xlabel('seconds (one core)')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('error of S   (S / S_true - 1)')
    axes[0].set_ylim(top=1e-2)          # SOR runs start far above this; errors over 1% are not of interest

    if pairs:
        # Third panel: for each method, its cheapest rung at the model's accuracy (filled marker: first S)
        # and the same plus one convergence check, every element size halved (open marker).
        ax = axes[2]
        ys = np.arange(len(pairs))[::-1]
        for y, (m, name, color, marker, base, chk) in zip(ys, pairs):
            t0, t1 = base['first'], base['first'] + chk['first']
            big = 110 if marker == '*' else 36
            ax.plot([t0, t1], [y, y], color=color, lw=1.5, zorder=2)
            ax.scatter([t0], [y], color=color, marker=marker, s=big, zorder=3, edgecolors='white', linewidths=0.8)
            ax.scatter([t1], [y], facecolors='white', edgecolors=color, marker=marker, s=big, zorder=3,
                       linewidths=1.3)
            ax.text(t1 * 1.25, y, f'x{t1 / t0:.2g}', color=muted, va='center', ha='left', fontsize=8)
        ref = chosen_run(rows, MODEL)
        if ref is not None:
            ax.axvline(ref['first'], color=muted, lw=1.0, ls='--', zorder=0)
            ax.text(ref['first'] * 1.08, 0.01, "model's first S", color=muted, ha='left', va='bottom',
                    transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set_yticks(ys)
        ax.set_yticklabels([name + (' *' if m == MODEL else '') for m, name, *_ in pairs], fontsize=8)
        ax.set_ylim(-0.7, len(pairs) - 0.3)
        ax.set_xscale('log')
        lo = min(b['first'] for *_, b, c in pairs)
        hi = max(b['first'] + c['first'] for *_, b, c in pairs)
        ax.set_xlim(lo / 2, hi * 4)
        ax.grid(True, which='major', axis='x', color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title('First S at the model\'s accuracy, and + one check (open)', fontsize=10, color=ink, loc='left')
        ax.set_xlabel('seconds (one core)')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='y', length=0)
        note = 'check: the same method once more, every element size halved'
        if any(m == MODEL for m, *_ in pairs):
            note += "\n* the model's check (node spacing halved) sees only its discretization error"
        ax.text(0.0, -0.17, note, color=muted, fontsize=7.5, transform=ax.transAxes, ha='left', va='top')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4 if pairs else 3, frameon=False, fontsize=8.5)
    cpus = sorted({r['cpu'] for r in rows if r.get('cpu')})            # the machine(s) of the logged runs
    cpu = cpus[0] if len(cpus) == 1 else ('several machines' if cpus else bc.machine_info()['cpu'])
    fig.suptitle(f"IDE test problem: error versus computing time   ({cpu}; hollow markers: below the true S)",
                 fontsize=9.5, color=ink, x=0.01, ha='left')
    fig.savefig(path_png, dpi=200)
    fig.savefig(os.path.splitext(path_png)[0] + '.pdf')
    print(f'Figure: {path_png} (and .pdf)')


def main():
    p = argparse.ArgumentParser(description='Comparison figure and summary table')
    p.add_argument('--csv', default=bc.RESULTS_CSV)
    p.add_argument('--trajectories', default=os.path.join(bc.RESULTS_DIR, 'sor_trajectories.csv'))
    a = p.parse_args()
    rows, checks = load(a.csv)
    text = summary(rows, checks)
    print(text)
    os.makedirs(bc.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(bc.RESULTS_DIR, 'summary.txt'), 'w') as f:
        f.write(text + '\n')
    try:
        figure(rows, checks, trajectories(a.trajectories), os.path.join(bc.RESULTS_DIR, 'benchmark_comparison.png'))
    except ImportError:
        print('matplotlib is not installed (pip install matplotlib): table only')


if __name__ == '__main__':
    main()
