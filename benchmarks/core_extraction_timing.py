r"""
core_extraction_timing.py -- how long the model takes to extract the dielectric functions of the paper's
blanks and films, and how long the numerical methods of Table S1 would take for the same work.

THE SCOPE.  Only the extraction itself: at each frequency, find the dielectric constant
for which the computed capacitance equals the measured one.  For the two blanks and the 13 film measurements,
at the 78 frequencies from 111 Hz to 9.67 MHz, exactly as reproduce_paper.py does it (Sec. 3.2.7):
    blank  eps_Q from Cp at every frequency, and from Cp - iG/omega (complex) below 2 MHz;
    film   eps_P from Cp at every frequency, and from Cp - iG/omega below 2 MHz (the paper's eps_r there;
           the Cp-only value is Newton's starting point).
Each blank's substrate spectrum is computed once and used for every film on that batch (extract() recomputes
it, at all 101 frequencies of the blank file, for every film).  Left out:
    the water cell;
    the uncertainty budget (uncertainty.py), which repeats the extraction with changed inputs;
    the Kramers-Kronig analysis (kramers_kronig.py).  Its fit uses eps_r only and costs the same for every
        method; its model evaluations are a repeat of this extraction (plus 5-100 Hz for the six films whose
        KK input starts at 5 Hz).
Variant 'Cp only': the same without the complex fits (real dielectric constants only).

WHAT THE SCRIPT DOES
1. times one capacitance (setup, one real S, one complex S; medians of repeats) on the benchmark problem of
   bench_common.py, for the six direct methods of Table S1: the model, the biquadratic FEM (tuned mesh), the
   bilinear bound average, fem2d.py on its tuned mesh and on the meshes of Supplemental Section S4 (t_e/32), and the five-point
   equations on the graded grid (direct).  Relaxation is left out: it was timed only from V = 0.
2. runs the extraction with the repository's own functions (extract_permittivity.substrate_spectrum and
   film_spectrum): first untimed-instrumentation runs for the wall time (median of 5), then once with every
   evaluation of IDEModel.S counted and timed, by kind (real/complex) and by geometry.  It checks that eps_r
   is identical to what extract_permittivity.extract() gives.
3. estimates each method:  (geometries) x setup + (real evaluations) x (one real S)
                                                + (complex evaluations) x (one complex S),
   one mesh per distinct (electrodes, film thickness), as whole_analysis_timing.py does.

Run (in this folder):   py core_extraction_timing.py        (under a minute)
Output: printed, and written to results\core_extraction.txt.
Written with Claude, Nov 2026.
"""
import bench_common as bc          # first: pins numpy/scipy to one thread, and finds the repository

import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'core_extraction.txt')
FMIN, FMAX, CUTOFF = 100.0, 1e7, 2e6          # as reproduce_paper.py / extract_permittivity.extract()
E, Q = bc.EPS_FILM, bc.EPS_SUB
EC, QC = E - 5.0j, Q - 0.01j                   # typical losses (the times do not depend on the values)


# ---- 1. one capacitance with each direct method ---------------------------------------------------------
def ft_S_any(cell, eps_film, eps_sub, form):
    """fem_tensor.QuarterCell.S for real or complex eps: same matrices, same SuperLU call (solve_spd),
    no complex conjugates (S continued analytically).  form = 'primal' (upper bound) or 'dual' (lower)."""
    import fem_tensor as ft
    if form == 'primal':
        c = {'sub': eps_sub, 'film': eps_film, 'air': 1.0}
        fixed, u_bc = cell.fixed_V, cell.V_bc
    else:
        c = {'sub': 1.0 / eps_sub, 'film': 1.0 / eps_film, 'air': 1.0}
        fixed, u_bc = cell.fixed_psi, cell.psi_bc
    K = sum(c[r] * cell.K[r] for r in cell.K)
    A, rhs = ft.reduce_to_free(K, fixed, u_bc)
    u = u_bc.astype(complex)
    u[~fixed], _ = ft.solve_spd(A, rhs)
    coef = sum(c[r] * cell.region[r] for r in cell.region)
    W = cell.mesh.energy(u, coef)
    return 4.0 / np.pi * W if form == 'primal' else 4.0 / np.pi / W


def method_times(im, fem2d):
    """Setup, one real S and one complex S for the six direct methods.  Returns a list of dicts."""
    import whole_analysis_timing as wat
    import fem_tensor as ft
    import scipy.sparse.linalg as spla
    rows = wat.one_S_times(im, fem2d)               # model, biquadratic FEM, fem2d.py t_e/32
    model, biq, appc = rows
    model['label'], biq['label'], appc['label'] = ('Our model', 'FEM, biquadratic, tuned mesh',
                                                   'FEM, bilinear, meshes of Section S4 (t_e/32)')
    g = im.IDEGeometry(s_um=bc.S_GAP, w_um=bc.W_FINGER, te_um=bc.T_E)

    # bilinear upper and lower bound and their average (bench_fem_bounds_average.py, first rung)
    geom = dict(s=bc.S_GAP, w=bc.W_FINGER, te=bc.T_E, tp=bc.T_P, y_extent_in_a=bc.Y_EXTENT_IN_A)
    cell, t_setup = bc.median_time(lambda: ft.QuarterCell(p=1, hc=bc.T_E / 1, growth=0.6, hx_max=0.5,
                                                          hy_max=4.0, **geom), 7)
    S_r, t_r = bc.median_time(lambda: 0.5 * (cell.S(E, Q, 'primal') + cell.S(E, Q, 'dual')), 21)
    S_c, t_c = bc.median_time(lambda: 0.5 * (ft_S_any(cell, EC, QC, 'primal') + ft_S_any(cell, EC, QC, 'dual')), 21)
    bavg = dict(label='FEM, bilinear, average of upper and lower bounds', nodes=cell.n_nodes, setup=t_setup,
                real=t_r, complex=t_c, S=S_r, S_cx=S_c)
    del cell

    # fem2d.py on its tuned mesh (bench_fem2d_cheapest.py)
    fem, t_setup = bc.median_time(lambda: fem2d.FEM2D(g, bc.T_P, hc_um=bc.T_E / 6, growth=0.06,
                                                      hx_max_um=2.0, hy_max_um=16.0), 3)
    S_r, t_r = bc.median_time(lambda: fem.S(E, Q), 5)
    S_c, t_c = bc.median_time(lambda: fem.S(EC, QC), 5)
    tuned = dict(label='FEM, bilinear, tuned mesh (fem2d.py)', nodes=fem.n_nodes, setup=t_setup,
                 real=t_r, complex=t_c, S=S_r, S_cx=S_c)
    del fem

    # five-point equations, graded grid refine 3, SuperLU (bench_direct_graded.py)
    import fivepoint as fp
    fp.warm_up()

    def fd_setup():
        xs, ys = fp.ladder_grid(3)
        wx, wy, fixed, V = fp.setup(xs, ys)
        Kff, rhs, free = fp.free_matrix(wx, wy, fixed, V)
        return xs, ys, wx, wy, fixed, V, Kff, rhs, free
    (xs, ys, wx, wy, fixed, V, Kff, rhs, free), t_setup = bc.median_time(fd_setup, 3)

    def fd_real():
        Vs = V.copy()
        Vs.ravel()[free] = spla.splu(Kff, permc_spec='MMD_AT_PLUS_A').solve(rhs)
        return fp.energy_S(Vs, wx, wy)
    # complex bond weights: they are affine in (eps_film, eps_sub), so three real setups give them exactly
    a, b, c = fp.setup(xs, ys, 0.0, 0.0), fp.setup(xs, ys, 1.0, 0.0), fp.setup(xs, ys, 0.0, 1.0)
    wxc, wyc = [a[k] + EC * (b[k] - a[k]) + QC * (c[k] - a[k]) for k in (0, 1)]
    Vc0 = V.astype(complex)
    Kc, rhsc, _ = fp.free_matrix(wxc, wyc, fixed, Vc0)

    def fd_complex():
        Vs = Vc0.copy()
        Vs.ravel()[free] = spla.splu(Kc, permc_spec='MMD_AT_PLUS_A').solve(rhsc)
        dx, dy = np.diff(Vs, axis=1), np.diff(Vs, axis=0)
        return 4.0 / math.pi * (np.sum(wxc * dx * dx) + np.sum(wyc * dy * dy))
    S_r, t_r = bc.median_time(fd_real, 5)
    S_c, t_c = bc.median_time(fd_complex, 5)
    fd = dict(label='Finite differences, graded grid, direct solution', nodes=V.size, setup=t_setup,
              real=t_r, complex=t_c, S=S_r, S_cx=S_c)
    return [model, biq, bavg, tuned, appc, fd]


# ---- 2. the extraction ---------------------------------------------------------------------------------
def core_extraction(ep, rp, im, cutoff=CUTOFF):
    """The two blanks and the 13 film measurements (see the module docstring).  Returns {label: eps_r}."""
    geoms = {'main': im.PAPER_MAIN_BATCH, 'second': im.PAPER_SECOND_BATCH}
    blanks = {'main': 'blank_main_batch.csv', 'second': 'blank_second_batch.csv'}
    models, subs, out = {}, {}, {}
    for batch, blank in blanks.items():
        model = models[batch] = im.IDEModel(geoms[batch])
        fb, Gb, Cb = ep.read_impedance_file(os.path.join(rp.DATA, blank))
        keep = (fb >= FMIN) & (fb <= FMAX)
        fb, Gb, Cb = fb[keep], Gb[keep], Cb[keep]
        sub = ep.substrate_spectrum(model, fb, Gb, Cb, cutoff)
        subs[batch] = (fb, sub)
        out['blank, ' + batch + ' batch'] = sub['eps_Cp']
    for smp in rp.read_samples():
        if smp['thickness_nm'] == 'bulk':
            continue
        model = models[smp['batch']]
        fb, sub = subs[smp['batch']]
        ff, Gf, Cf = ep.read_impedance_file(os.path.join(rp.DATA, smp['file']))
        keep = (ff >= FMIN) & (ff <= FMAX)
        ff, Gf, Cf = ff[keep], Gf[keep], Cf[keep]
        eq_cp = ep._interp_logf(ff, fb, sub['eps_Cp'])
        eq_cx = ep._interp_logf(ff, fb, sub['eps_complex'])
        film = ep.film_spectrum(model, ff, Gf, Cf, eq_cp, eq_cx, float(smp['thickness_nm']) / 1000.0, cutoff)
        out[smp['label']] = film['eps_r']
    return out


class Tally:
    """Evaluations of IDEModel.S during one run, by (kind, geometry)."""
    def __init__(self):
        self.setups, self.t_setup = 0, 0.0
        self.n, self.t = defaultdict(int), defaultdict(float)
        self.distinct = set()

    def total(self, kind):
        return sum(v for (k, _), v in self.n.items() if k == kind)

    def time(self, kind=None):
        return sum(v for (k, _), v in self.t.items() if kind in (None, k))

    def geometries(self):
        return {g for _, g in self.n}


TALLY = None


def instrument(im):
    """Wrap IDEModel.__init__ and IDEModel.S so that they fill TALLY (when it is set)."""
    cls = im.IDEModel
    orig_init, orig_S = cls.__init__, cls.S

    def init(self, *args, **kwargs):
        t0 = time.perf_counter()
        orig_init(self, *args, **kwargs)
        if TALLY is not None:
            TALLY.t_setup += time.perf_counter() - t0
            TALLY.setups += 1

    def S(self, eps_film, eps_sub, t_film_um, return_profile=False):
        t0 = time.perf_counter()
        out = orig_S(self, eps_film, eps_sub, t_film_um, return_profile)
        dt = time.perf_counter() - t0
        if TALLY is not None:
            kind = 'complex' if (np.iscomplexobj(eps_film) or np.iscomplexobj(eps_sub)) else 'real'
            geom = (self.s, self.w, self.te, round(float(t_film_um), 9))
            TALLY.n[(kind, geom)] += 1
            TALLY.t[(kind, geom)] += dt
            TALLY.distinct.add((kind,) + geom + (complex(eps_film), complex(eps_sub)))
        return out
    cls.__init__, cls.S = init, S


# ---- 3. report -----------------------------------------------------------------------------------------
def fmt(t):
    if t < 1e-3:
        return f'{t * 1e6:.0f} us'
    if t < 1:
        return f'{t * 1e3:.3g} ms'
    if t < 120:
        return f'{t:.3g} s'
    if t < 7200:
        return f'{t / 60:.3g} min'
    return f'{t / 3600:.3g} h'


def main():
    global TALLY
    if bc.REPO is None:
        sys.exit('ide_model.py not found: set the environment variable DIELECTRIC_IDES to the repository folder')
    import ide_model as im
    import fem2d
    import extract_permittivity as ep
    import reproduce_paper as rp
    m = bc.machine_info()
    lines = [f"core_extraction_timing.py  {time.strftime('%Y-%m-%d %H:%M')}",
             f"{m['cpu']} | {m['os']} | Python {m['python']}, numpy {m['numpy']}, scipy {m['scipy']} | "
             f"threads {m['threads']}", f'repository: {bc.REPO}', '']

    def say(text=''):
        print(text, flush=True)
        lines.append(text)
    print('\n'.join(lines), flush=True)

    print('timing one capacitance with each method ...', flush=True)
    rows = method_times(im, fem2d)

    # wall time of the extraction, before any instrumentation (median of 5 runs)
    print('timing the extraction ...', flush=True)
    walls = {}
    for name, cutoff in (('paper', CUTOFF), ('Cp only', 0.0)):
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = core_extraction(ep, rp, im, cutoff)
            ts.append(time.perf_counter() - t0)
        walls[name] = (float(np.median(ts)), res)

    # identical to extract()?
    geoms = {'main': im.PAPER_MAIN_BATCH, 'second': im.PAPER_SECOND_BATCH}
    worst = 0.0
    res = walls['paper'][1]
    for smp in rp.read_samples():
        if smp['thickness_nm'] == 'bulk':
            continue
        ref = ep.extract(os.path.join(rp.DATA, smp['blank_file']), os.path.join(rp.DATA, smp['file']),
                         float(smp['thickness_nm']), geom=geoms[smp['batch']], loss_cutoff_hz=CUTOFF)['eps_r']
        worst = max(worst, float(np.nanmax(np.abs(res[smp['label']] / ref - 1))))

    # counted runs
    instrument(im)
    tallies = {}
    for name, cutoff in (('paper', CUTOFF), ('Cp only', 0.0)):
        TALLY = Tally()
        core_extraction(ep, rp, im, cutoff)
        tallies[name] = TALLY
    TALLY = None

    say('1. THE EXTRACTION: 2 blanks and 13 film measurements, 78 frequencies (111 Hz - 9.67 MHz)')
    say(f'   eps_r agrees with extract_permittivity.extract() to {worst:.0e} (relative)')
    for name, t in tallies.items():
        R, C = t.total('real'), t.total('complex')
        say(f"   {name:8s}: {R:,d} real + {C:,d} complex = {R + C:,d} evaluations of S ({len(t.distinct):,d} distinct), "
            f"{len(t.geometries())} geometries, {t.setups} model setups; wall {fmt(walls[name][0])} (median of 5), "
            f"in S {fmt(t.time())}")
    t = tallies['paper']
    say('   the model per evaluation in this extraction, by geometry (paper variant):')
    for geom in sorted(t.geometries(), key=lambda g: (g[0], g[3])):
        parts = []
        for kind in ('real', 'complex'):
            n = t.n.get((kind, geom), 0)
            if n:
                parts.append(f'{kind} {n:5,d} x {fmt(t.t[(kind, geom)] / n):>6s}')
        label = 'blank' if geom[3] <= geom[2] + 1e-9 else f't = {geom[3] * 1000:.0f} nm'
        say(f'     s = {geom[0]:.2f} um, {label:12s} ' + '   '.join(parts))
    say()

    say('2. ONE CAPACITANCE ON THE BENCHMARK PROBLEM (medians of repeats)')
    say(f"   {'method':52s} {'nodes':>8s} {'setup':>9s} {'real S':>9s} {'complex S':>10s} {'cx/real':>7s}  S real")
    for r in rows:
        say(f"   {r['label']:52s} {r['nodes']:8,d} {fmt(r['setup']):>9s} {fmt(r['real']):>9s} {fmt(r['complex']):>10s} "
            f"{r['complex'] / r['real']:7.2f}  {r['S']:.9f}")
    say()

    say('3. ESTIMATED TIME FOR THE EXTRACTION')
    say('   each method: one setup per geometry + one solve per evaluation, at its benchmark-problem times')
    say(f"   {'method':52s} {'paper':>10s} {'x model':>8s} {'Cp only':>10s} {'x model':>8s}")
    meas = {name: walls[name][0] for name in walls}
    est = {}
    for r in rows:
        for name, t in tallies.items():
            n_set = t.setups if r is rows[0] else len(t.geometries())
            est[(r['label'], name)] = n_set * r['setup'] + t.total('real') * r['real'] + t.total('complex') * r['complex']
    for i, r in enumerate(rows):
        cells = []
        for name in tallies:
            v = est[(r['label'], name)]
            cells.append(f'{fmt(v):>10s} {v / meas[name]:8.0f}')
        label = r['label'] + (' (at benchmark times)' if i == 0 else '')
        say(f"   {label:52s} " + ' '.join(cells))
    say(f"   {'Our model, measured (wall time, median of 5)':52s} {fmt(meas['paper']):>10s} {1:8d} "
        f"{fmt(meas['Cp only']):>10s} {1:8d}")
    say("   'x model' = estimate / the model's measured wall time.  Against the model costed at benchmark times,")
    say('   multiply by ' + ', '.join(f"{meas[n] / est[(rows[0]['label'], n)]:.2f} ({n})" for n in tallies) + '.')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    main()
