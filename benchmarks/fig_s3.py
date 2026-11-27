"""
fig_s3.py -- Supplemental Figure S3 of the paper: error of the capacitance of the test problem against
computing time, for every mesh of every method in Supplemental Table S1, from the benchmark log
results/benchmark_results.csv (our desktop PC, Intel Core Ultra 7 265).

    py fig_s3.py                          (reads and writes this folder's results/; a few seconds)
    py fig_s3.py <results_dir> <out_dir>

Writes Fig_S3_benchmarks.png (300 dpi) and Fig_S3_benchmarks.pdf.  Needs numpy, pandas and matplotlib;
the paper's figure was drawn with the Liberation Sans font (matplotlib substitutes another font, with a
warning, if it is not installed).

(a) time for the first capacitance (mesh and matrices built, one solve); (b) time for each further
capacitance (mesh and matrices reused).  The dashed blue line is the error of our model with its
default settings.

Which runs are plotted:
  * direct methods and the model: every rung of the method's newest run (rows within 2 h of its
    newest row), checks and the legacy model setting left out;
  * relaxation (SOR): the newest row of each rung, rungs in order of size up to the first that reaches
    the model's accuracy (SOR stops as soon as S <= S_MODEL, so finer rungs only add time).
Errors are relative to bench_common.S_REF; |error| on the log axis, open symbols where S < S_REF.
Written with Claude, Nov 2026.
"""
import sys, os
LABEL_A_X = float(os.environ.get('LABEL_A_X', '0.985'))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results')
OUT = sys.argv[2] if len(sys.argv) > 2 else RES
os.makedirs(OUT, exist_ok=True)

S_REF = 7.18810231545            # bench_common.S_REF (biquadratic bounds on 4.7 M nodes, width 2e-10)
plt.rcParams.update({'font.family': 'Liberation Sans', 'mathtext.fontset': 'stix', 'font.size': 8,
                     'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.5,
                     'ytick.minor.width': 0.5, 'legend.frameon': False, 'pdf.fonttype': 42})

d = pd.read_csv(os.path.join(RES, 'benchmark_results.csv'))
d['t'] = pd.to_datetime(d.timestamp)
d['err'] = d.S / S_REF - 1.0
d = d[~d.label.str.startswith('check')]


def newest_run(method, exclude=()):
    r = d[d.method == method]
    for e in exclude:
        r = r[~r.label.str.startswith(e)]
    r = r[r.t >= r.t.max() - pd.Timedelta(hours=2)]
    return r.sort_values('n_nodes')


def sor_ladder(method):
    r = d[d.method == method].sort_values('t').groupby('label').tail(1).sort_values('n_nodes')
    hit = np.flatnonzero(r.beats_model.values == 1)
    return r.iloc[:hit[0] + 1] if len(hit) else r


BLUE, RED, GREEN, GRAY = '#1b6fd1', '#d1491b', '#009e73', '#555555'
# (label, colour, marker, marker size, line style, rows)
SERIES = [        # same order and names as Table S1
    ('Our model', BLUE, 'o', 3.6, '-', newest_run('model (ide_model.py)', exclude=('legacy',))),
    ('FEM, biquadratic elements', RED, 'D', 3.2, '-', newest_run('FEM biquadratic (upper bound)')),
    ('FEM, bilinear elements, average of upper and lower bounds', GREEN, '^', 3.8, (0, (4, 2)),
     newest_run('FEM bound average (bilinear)')),
    ('FEM, bilinear elements, mesh tuned to this problem', GREEN, '*', 6.5, '', newest_run('fem2d.py, tuned mesh')),
    ('FEM, bilinear elements, meshes of Section S4', GREEN, 's', 3.4, '-', newest_run('fem2d.py (bilinear, direct)')),
    ('Finite differences, graded grid, direct solution', GRAY, 'v', 3.8, '-',
     newest_run('five-point, direct (graded grid)')),
    ('Finite differences, graded grid, relaxation (SOR)', GRAY, 'P', 4.0, (0, (1, 1.3)),
     sor_ladder('five-point, SOR (graded grid)')),
    ('Finite differences, uniform grid, relaxation (SOR)', GRAY, 'X', 4.0, (0, (5, 1.5, 1, 1.5)),
     sor_ladder('SOR, uniform squares')),
]
MODEL_ERR = SERIES[0][5].iloc[0].err          # default settings (43 in-plane nodes): the first row

# about 5.85 in after trimming, shown 6.3 in wide in the SI: fonts then appear at 8.6 pt, as in Figs. S3-S9
fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.0))
fig.subplots_adjust(left=0.09, right=0.985, top=0.945, bottom=0.345, wspace=0.29)
for ax, col in zip(axs, ('total_s', 'per_eval_s')):
    ax.axhline(MODEL_ERR, color=BLUE, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for lab, c, mk, ms, ls, r in SERIES:
        r = r[np.isfinite(r[col])]
        if not len(r):
            continue
        x, e = r[col].values, r.err.values
        if ls != '':
            ax.plot(x, np.abs(e), ls=ls, color=c, lw=0.9, zorder=2)
        neg = e < 0
        ax.plot(x[~neg], np.abs(e[~neg]), ls='', marker=mk, ms=ms, color=c, mec=c, mew=0.5, zorder=3)
        if neg.any():
            ax.plot(x[neg], np.abs(e[neg]), ls='', marker=mk, ms=ms, mfc='white', mec=c, mew=0.8, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(1e-9, 3e-3)
    ax.set_ylabel('relative error of the capacitance')
    ax.tick_params(which='both', top=False, right=False)
axs[0].set_xlabel('time for the first capacitance (s)')
axs[1].set_xlabel('time for each further capacitance (s)')
xmax0 = max(r.total_s.max() for *_, r in SERIES)
axs[0].set_xlim(3e-4, 10 ** np.ceil(np.log10(xmax0 * 1.5)))
axs[1].set_xlim(3e-5, 30)
for ax, l in zip(axs, 'ab'):
    ax.text(-0.17, 1.0, f'({l})', transform=ax.transAxes, fontsize=9, fontweight='bold', ha='left', va='bottom')
# labels of the dashed line (blue, like our model's points): panel (b) at the right end above the line
axs[1].text(0.985, MODEL_ERR * 1.35, "our model's error", transform=axs[1].get_yaxis_transform(),
            ha='right', va='bottom', fontsize=6.5, color=BLUE)
# panel (a): the right end of the line has the uniform-grid SOR points on it, so the label goes below the line
axs[0].text(LABEL_A_X, MODEL_ERR / 1.35, "our model's error", transform=axs[0].get_yaxis_transform(),
            ha='right', va='top', fontsize=6.5, color=BLUE)

handles = [Line2D([], [], color=c, ls=(ls if ls != '' else 'none'), lw=0.9, marker=mk, ms=ms, mec=c, label=lab)
           for lab, c, mk, ms, ls, _ in SERIES]
handles.append(Line2D([], [], ls='none', marker='o', ms=3.6, mfc='white', mec='0.3', mew=0.8,
                      label='open symbols: below the reference value'))
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=6.5,      # columns: FEM | finite differences
           bbox_to_anchor=(0.535, 0.0), columnspacing=2.5, handlelength=3.2, handletextpad=0.6,
           labelspacing=0.4, borderaxespad=0.2)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'Fig_S3_benchmarks.{ext}'), dpi=300, bbox_inches='tight', pad_inches=0.02)
print('model error', MODEL_ERR)
for lab, c, mk, ms, ls, r in SERIES:
    print(f'{lab:60s} {len(r):2d} rungs  newest {r.timestamp.max()}  oldest {r.timestamp.min()}')
