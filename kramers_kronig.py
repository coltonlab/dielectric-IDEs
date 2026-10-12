"""
kramers_kronig.py
=================

Imaginary part eps_i(f) of the dielectric function from the measured real part eps_r(f), by a
Kramers-Kronig (KK) constrained fit (Appendix B of McClure et al. 2026, after A. B. Kuzmenko,
Rev. Sci. Instrum. 76, 083108 (2005)).

    python kramers_kronig.py --input MAPI_s1.csv --out MAPI_s1_kk.csv --plot

The input is the CSV file written by extract_permittivity.py (it uses the columns f_Hz and eps_r),
or any CSV file with those two column names.

    python kramers_kronig.py --paper      (eps_i of all films of the paper, about 7 minutes)


THE METHOD
----------
eps_i(f) is written as a sum of non-negative triangular ("hat") functions on a logarithmic grid of
nodes: every second measured frequency, extended 2 decades below and 1 decade above the data, so
that losses outside the measured range can contribute.  Each triangle's KK partner is known
analytically, so

    eps_r(f) = eps_inf + sum_k A_k g_k(f),       A_k >= 0,  eps_inf >= 1,

is a linear least-squares problem for the amplitudes A_k (the eps_i values at the nodes).  A small
penalty on the curvature of A_k keeps the solution smooth (strength lam, default 0.1).  Each point
is weighted by 1/(0.1% of eps_r).  A DC conductivity contributes to the measured loss but not to
eps_r, so this eps_i excludes conduction losses.

WHERE eps_i IS DETERMINED
-------------------------
Near either end of the fitted range eps_i is NOT determined by the data: losses just outside the
range can be traded against each other.  As in the paper, eps_i is flagged as determined where it
changes by less than 20% both when the curvature penalty is changed over a factor of about 30
(lam = 0.03 to 1) and when the fitted range is shortened by two frequencies at each end.  The column
eps_i_determined is 1 there and 0 elsewhere; the paper draws the undetermined parts dashed.

INPUT RANGE
-----------
Use only frequencies where eps_r is reliable.  For our data that is 111 Hz to 9.67 MHz.  For the
three films whose capacitance at low frequency is far above the instrumental rise of the blank
(PEA2PbI4, BA2PbI4, HA2PbI4), the paper extends the input down to 5 Hz with a lossless substrate
below 100 Hz (extract_permittivity.py --fmin 0 --substrate-fixed-below 90); --paper does this.
"""

import argparse
import csv
import os

import numpy as np
from scipy.optimize import lsq_linear


# --------------------------------------------------------------------------------------------
# Analytic KK transform of one linear segment, and of a triangular basis function
# --------------------------------------------------------------------------------------------
def _F(x, w, al, be):
    """Antiderivative used for the KK transform of eps_i = al + be*x on a segment."""
    lm = np.log(np.abs(x - w) + 1e-300)
    lp = np.log(x + w)
    return 0.5 * (al + be * w) * lm + 0.5 * (al - be * w) * lp + be * x


def _segment(w, a, b, fa, fb):
    """KK partner (contribution to eps_r at frequency w) of eps_i rising linearly from fa at a to
    fb at b:  eps_r(w) = (2/pi) P int_a^b x eps_i(x) / (x^2 - w^2) dx."""
    be = (fb - fa) / (b - a)
    al = fa - be * a
    return (2 / np.pi) * (_F(b, w, al, be) - _F(a, w, al, be))


def kk_design_matrix(f, nodes):
    """Matrix whose column k is the KK partner g_k(f) of the triangle of height 1 at nodes[k]."""
    cols = []
    for k in range(len(nodes)):
        g = np.zeros_like(f, dtype=float)
        if k > 0:
            g += _segment(f, nodes[k - 1], nodes[k], 0.0, 1.0)
        if k < len(nodes) - 1:
            g += _segment(f, nodes[k], nodes[k + 1], 1.0, 0.0)
        cols.append(g)
    return np.column_stack(cols)


def make_nodes(f, step=2, decades_below=2.0, decades_above=1.0):
    """Nodes: every `step`-th data frequency, extended on the same logarithmic spacing by
    `decades_below` decades below the data and `decades_above` decades above."""
    nd = list(f[::step])
    if nd[-1] < f[-1] * 0.999:
        nd.append(f[-1])
    lr = np.log(f[step] / f[0])
    nlo = int(round(decades_below * np.log(10) / lr))
    nhi = int(round(decades_above * np.log(10) / lr))
    lo = [nd[0] * np.exp(-lr * i) for i in range(nlo, 0, -1)]
    hi = [nd[-1] * np.exp(lr * i) for i in range(1, nhi + 1)]
    return np.array(lo + nd + hi)


def fit_kk(f, eps_r, nodes, lam=0.1, rel=1e-3, einf_min=1.0):
    """Least-squares KK fit of eps_r(f).  Returns a dict with the node amplitudes, eps_inf, a function
    eps_i(f) (linear interpolation between nodes) and the rms relative residual."""
    T = kk_design_matrix(f, nodes)
    nK = len(nodes)
    sig = rel * np.abs(eps_r)                                 # each point weighted by 1/(0.1% of eps_r)
    A = np.hstack([np.ones((len(f), 1)), T]) / sig[:, None]
    b = eps_r / sig
    # curvature penalty on A_k / s_k, where s_k is the local scale of eps_r (so that the penalty works
    # the same way for eps_r of 5 and of 5000)
    s = np.interp(np.log(nodes), np.log(f), np.abs(eps_r))
    D = np.zeros((nK - 2, 1 + nK))
    for i in range(nK - 2):
        D[i, 1 + i:1 + i + 3] = np.array([1, -2, 1]) / s[i:i + 3]
    A2 = np.vstack([A, np.sqrt(lam) * D / rel])
    b2 = np.concatenate([b, np.zeros(nK - 2)])
    lower = np.full(1 + nK, 0.0)                              # A_k >= 0 (losses are positive)
    lower[0] = einf_min                                       # eps_inf >= 1
    r = lsq_linear(A2, b2, bounds=(lower, np.full(1 + nK, np.inf)), method='bvls', max_iter=20000)
    p = r.x
    resid = np.hstack([np.ones((len(f), 1)), T]) @ p - eps_r
    return dict(amplitudes=p[1:], eps_inf=p[0], nodes=nodes,
                eps_i=lambda x: np.interp(x, nodes, p[1:], left=0, right=0),
                eps_r_fit=eps_r + resid, rms_rel=np.sqrt(np.mean((resid / np.abs(eps_r)) ** 2)))


def kk_analysis(f, eps_r, lam=0.1, lam_range=(0.03, 0.1, 0.3, 1.0), shorten=2, tolerance=0.2):
    """eps_i at the frequencies f from eps_r(f), with the determinacy test of Appendix B.
    Returns a dict: eps_i (the fit with lam), eps_i_low/high (range over all variants), determined
    (bool array), rms_rel and eps_r_fit."""
    ok = np.isfinite(eps_r) & (eps_r > 0)
    ff, ee = f[ok], eps_r[ok]
    base = fit_kk(ff, ee, make_nodes(ff), lam)
    variants = [base['eps_i'](f)]
    for l in lam_range:                                       # change the smoothing
        if l != lam:
            variants.append(fit_kk(ff, ee, make_nodes(ff), l)['eps_i'](f))
    short = slice(shorten, len(ff) - shorten)                 # shorten the range at both ends
    variants.append(fit_kk(ff[short], ee[short], make_nodes(ff[short]), lam)['eps_i'](f))
    V = np.array(variants)
    e2 = V[0]
    lo, hi = V.min(0), V.max(0)
    determined = (e2 > 0) & ((hi - lo) <= tolerance * np.maximum(e2, 1e-3))
    fit_full = np.full(len(f), np.nan)
    fit_full[ok] = base['eps_r_fit']
    return dict(eps_i=e2, eps_i_low=lo, eps_i_high=hi, determined=determined, rms_rel=base['rms_rel'],
                eps_r_fit=fit_full, eps_inf=base['eps_inf'])


# --------------------------------------------------------------------------------------------
# Input / output
# --------------------------------------------------------------------------------------------
def read_eps_csv(path, f_col='f_Hz', eps_col='eps_r'):
    with open(path, newline='') as fh:
        rows = [l for l in fh if not l.startswith('#')]
    r = list(csv.DictReader(rows))
    f = np.array([float(x[f_col]) for x in r])
    e = np.array([float(x[eps_col]) if x[eps_col] not in ('', 'nan') else np.nan for x in r])
    return f, e


def write_kk_csv(path, f, eps_r, res):
    with open(path, 'w', newline='') as fh:
        fh.write(f'# Kramers-Kronig analysis (Appendix B); rms residual of the fit {100 * res["rms_rel"]:.3f}%; '
                 f'eps_inf = {res["eps_inf"]:.4g}\n')
        fh.write('# eps_i_low/high = range over the smoothing and range variants; eps_i_determined = 1 where the '
                 'variants agree to 20%\n')
        w = csv.writer(fh)
        w.writerow(['f_Hz', 'eps_r', 'eps_r_KK_fit', 'eps_i', 'eps_i_low', 'eps_i_high', 'eps_i_determined'])
        for j in range(len(f)):
            w.writerow([f'{f[j]:.6g}', f'{eps_r[j]:.6g}', f'{res["eps_r_fit"][j]:.6g}', f'{res["eps_i"][j]:.5g}',
                        f'{res["eps_i_low"][j]:.5g}', f'{res["eps_i_high"][j]:.5g}', int(res['determined'][j])])


def plot_kk(path_png, f, eps_r, res, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    ax.loglog(f, eps_r, 'k-', label='eps_r (input)')
    e2 = np.where(res['eps_i'] > 0, res['eps_i'], np.nan)
    ax.loglog(f, e2, 'r--', lw=0.8, label='eps_i, not determined')
    ax.loglog(f, np.where(res['determined'], e2, np.nan), 'r-', label='eps_i (KK), determined')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Dielectric function')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path_png)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Reproduce the paper
# --------------------------------------------------------------------------------------------
def paper(only=None):
    """eps_i of every film of the paper (or only the samples whose labels are listed in `only`).
    Writes results/<label>_KK.csv for each run, then prints the compound values at 1.06 MHz
    (Sec. IV B) from all the _KK.csv files present."""
    import extract_permittivity as ep
    import reproduce_paper as rp
    from ide_model import PAPER_MAIN_BATCH, PAPER_SECOND_BATCH
    geoms = {'main': PAPER_MAIN_BATCH, 'second': PAPER_SECOND_BATCH}
    os.makedirs(rp.RESULTS, exist_ok=True)
    kkfile = lambda label: os.path.join(rp.RESULTS, label.replace(' ', '_') + '_KK.csv')
    for smp in rp.read_samples():
        if smp['thickness_nm'] == 'bulk' or (only and smp['label'] not in only):
            continue
        extended = smp['compound'] in ('PEA2PbI4', 'BA2PbI4', 'HA2PbI4')   # input from 5 Hz (Appendix B)
        res = ep.extract(os.path.join(rp.DATA, smp['blank_file']), os.path.join(rp.DATA, smp['file']),
                         float(smp['thickness_nm']), geom=geoms[smp['batch']],
                         fmin=0.0 if extended else 100.0, fmax=1e7,
                         substrate_fixed_below_hz=90.0 if extended else None)
        f, e1 = res['f_Hz'], res['eps_r']
        kk = kk_analysis(f, e1)
        keep = (f >= 100) & (f <= 1e7)                            # the range shown in the paper
        write_kk_csv(kkfile(smp['label']), f[keep], e1[keep],
                     {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in kk.items()})
        j = int(np.argmin(np.abs(np.log(f / 1.06e6))))
        print(f'{smp["label"]:12s} input {f[0]:7.1f} Hz - {f[-1] / 1e6:.2f} MHz, rms {100 * kk["rms_rel"]:.2f}%, '
              f'eps_i(1.06 MHz) = {kk["eps_i"][j]:.3f} (range {kk["eps_i_low"][j]:.3f}-{kk["eps_i_high"][j]:.3f}, '
              f'{"determined" if kk["determined"][j] else "NOT determined"})', flush=True)
    print('\nCompound eps_i at 1.06 MHz (paper, Sec. IV B: 1.4, 2.7, 4.7, 4.2, below 0.1, about 0.2; +-25%):')
    for comp, runs in rp.COMPOUND_RUNS.items():
        if not all(os.path.exists(kkfile(r)) for r in runs):
            print(f'  {comp:10s} (not all runs analyzed yet)')
            continue
        vals = []
        for r in runs:
            fr, _ = read_eps_csv(kkfile(r))
            _, e2 = read_eps_csv(kkfile(r), eps_col='eps_i')
            vals.append(e2[int(np.argmin(np.abs(np.log(fr / 1.06e6))))])
        print(f'  {comp:10s} {np.mean(vals):.2f}')


def main(argv=None):
    p = argparse.ArgumentParser(description='Kramers-Kronig analysis: eps_i from eps_r (Appendix B of McClure et al. 2026).')
    p.add_argument('--paper', action='store_true', help='analyze all films of the paper')
    p.add_argument('--only', help='with --paper: comma-separated sample labels to analyze (e.g. "MAPI s1,PEPI s8")')
    p.add_argument('--input', help='CSV with columns f_Hz and eps_r (e.g. written by extract_permittivity.py)')
    p.add_argument('--eps-column', default='eps_r', help='column of the input holding eps_r (default eps_r)')
    p.add_argument('--fmin', type=float, default=0.0, help='use only input frequencies >= fmin (Hz)')
    p.add_argument('--fmax', type=float, default=np.inf, help='use only input frequencies <= fmax (Hz)')
    p.add_argument('--lam', type=float, default=0.1, help='strength of the curvature penalty (default 0.1)')
    p.add_argument('--out', help='output CSV (default <input>_KK.csv)')
    p.add_argument('--plot', action='store_true', help='also save a PNG plot')
    a = p.parse_args(argv)
    if a.paper:
        paper([s.strip() for s in a.only.split(',')] if a.only else None)
        return
    if not a.input:
        p.error('give --input (or --paper)')
    f, e1 = read_eps_csv(a.input, eps_col=a.eps_column)
    keep = (f >= a.fmin) & (f <= a.fmax)
    f, e1 = f[keep], e1[keep]
    res = kk_analysis(f, e1, lam=a.lam)
    out = a.out or os.path.splitext(a.input)[0] + '_KK.csv'
    write_kk_csv(out, f, e1, res)
    if a.plot:
        plot_kk(os.path.splitext(out)[0] + '.png', f, e1, res, title=os.path.basename(a.input))
    j = int(np.argmin(np.abs(np.log(f / 1.06e6))))
    print(f'results written to {out}; fit rms {100 * res["rms_rel"]:.3f}%')
    print(f'eps_i at {f[j]:.4g} Hz = {res["eps_i"][j]:.4g} ({"determined" if res["determined"][j] else "NOT determined"})')


if __name__ == '__main__':
    main()
