r"""
whole_analysis_timing.py -- how long the paper's complete analysis takes with the model, and how long it
would take with a finite-element solver of the same accuracy.

Every number in the paper comes from three scripts of the repository: reproduce_paper.py (all the eps
values, from the raw data), uncertainty.py --paper (the uncertainty budgets of Sec. 5.1 and Supplemental Section S12) and
kramers_kronig.py --paper (eps_i, Supplemental Section S8).  This script

1. runs those three scripts on a temporary copy of the repository (the clone and its results\ folder are
   not touched) and counts what the model does in them:
     setups       IDEModel(...) constructions (extract_permittivity.extract() makes one per call);
     evaluations  calls of IDEModel.S, the capacitance itself, with real or with complex dielectric
                  constants.  Everything else (S_blank, solve_substrate, solve_film, the two complex
                  solvers, the finite-difference derivatives of the uncertainty budget) goes through S,
                  so this counts every capacitance the analysis computes;
     geometries   the distinct (electrodes, film thickness) combinations at which S is evaluated.  A
                  finite-element solver needs a new mesh for each of them; the model does not.
   It also times each script.

2. times one capacitance with real and with complex dielectric constants (setup reused; medians of
   repeats) on the benchmark problem of bench_common.py, for
     the model at its default settings (as bench_model.py),
     fem2d.py at the rung h_c = t_e/32, growth 0.06 (as bench_fem2d.py), and
     the biquadratic FEM of fem_tensor.py at the rung 'coarse: g 0.6, hc t_e/4, caps 6/4'
     (as bench_fem_quadratic.py),
   the cheapest rung of each that is at least as accurate as the model.  fem_tensor.QuarterCell.S works
   in real arithmetic only, so for complex eps this script repeats its steps with complex numbers (same
   matrices, same SuperLU settings): biquadratic_S_complex() below.

3. estimates the time each method would need for the whole analysis:
       (geometries) x setup  +  (real evaluations) x (one real S)  +  (complex evaluations) x (one complex S)
   replacing every evaluation of the model by one solve of the method, and counting the FEM setup (mesh
   and per-region matrices) once per distinct geometry.  The counts are those of this code: brentq for
   real eps, Newton's method with a central-difference derivative for complex eps (3 evaluations per
   step), and finite differences for the derivatives of the uncertainty budget.  A program built around
   a finite-element solver could need fewer solves (the derivative of S with respect to an eps follows
   from the field of the same solve, for example), so the FEM numbers are upper ends.  The line
   'distinct evaluations only' leaves out the evaluations that the three scripts repeat (each script
   recomputes the blank substrates, and every film is extracted in more than one script).
   Relaxation (SOR) gets no estimate: its time per S depends on the starting potential, and the
   benchmarks time it only from V = 0.

Run (in this folder):   py whole_analysis_timing.py        (about half a minute)
Output: printed, and written to results\whole_analysis.txt (the scripts' own output goes to
results\whole_analysis_log.txt).

Written with Claude, Nov 2026.
"""
import bench_common as bc          # first: pins numpy/scipy to one thread, and finds the repository

import contextlib
import io
import os
import runpy
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'whole_analysis.txt')
LOG = os.path.join(HERE, 'results', 'whole_analysis_log.txt')

# the three scripts behind the paper's numbers, with their command-line arguments
SCRIPTS = [('reproduce_paper.py', []), ('uncertainty.py', ['--paper']), ('kramers_kronig.py', ['--paper'])]
SOLVERS = ['solve_substrate', 'solve_film', 'solve_substrate_complex', 'solve_film_complex']

# the benchmark rungs at the model's accuracy (results\summary.txt)
FEM2D_RUNG = dict(hc_um=bc.T_E / 32, growth=0.06)            # fem2d.py: h_c = t_e/32, growth 0.06
BIQ_LABEL = 'coarse: g 0.6, hc t_e/4, caps 6/4'               # fem_tensor.py, p = 2
# typical losses for the complex timings (the times do not depend on the values)
EPS_FILM_CX, EPS_SUB_CX = bc.EPS_FILM - 5.0j, bc.EPS_SUB - 0.01j


# ---- 1. counting what the model does --------------------------------------------------------------
class Count:
    """What the model did during one script."""

    def __init__(self, name, args):
        self.label = ' '.join([name] + list(args))
        self.setups, self.t_setup = 0, 0.0
        self.n = {'real': 0, 'complex': 0}            # evaluations of S
        self.t = {'real': 0.0, 'complex': 0.0}        # time inside S
        self.geometries = set()                       # (s, w, t_e, t_film) at which S was evaluated
        self.distinct = set()                         # distinct evaluations: kind, geometry and eps values
        self.inversions = {k: 0 for k in SOLVERS}     # calls of each solver (outermost only)
        self.inside = {k: 0 for k in SOLVERS}         # evaluations of S made inside each solver
        self.failed = {k: 0 for k in SOLVERS}         # solver calls that raised an error (e.g. no root)
        self.wall = 0.0                               # wall time of the whole script


NOW = None          # the Count being filled
STACK = []          # the solvers currently running (to attribute each evaluation of S to one)
FIRST = set()       # every distinct evaluation seen so far, in any script ...
T_FIRST = {'real': 0.0, 'complex': 0.0}     # ... and the time of its first evaluation


def instrument(im):
    """Wrap IDEModel.__init__, IDEModel.S and the four solvers of the module ide_model (im) so that they
    fill NOW.  The wrapped functions call the originals unchanged."""
    cls = im.IDEModel
    orig_init, orig_S = cls.__init__, cls.S

    def init(self, *args, **kwargs):
        t0 = time.perf_counter()
        orig_init(self, *args, **kwargs)
        NOW.t_setup += time.perf_counter() - t0
        NOW.setups += 1

    def S(self, eps_film, eps_sub, t_film_um, return_profile=False):
        t0 = time.perf_counter()
        out = orig_S(self, eps_film, eps_sub, t_film_um, return_profile)
        dt = time.perf_counter() - t0
        kind = 'complex' if (np.iscomplexobj(eps_film) or np.iscomplexobj(eps_sub)) else 'real'
        NOW.n[kind] += 1
        NOW.t[kind] += dt
        geometry = (self.s, self.w, self.te, round(float(t_film_um), 9))
        NOW.geometries.add(geometry)
        key = (kind,) + geometry + (complex(eps_film), complex(eps_sub))
        NOW.distinct.add(key)
        if key not in FIRST:
            FIRST.add(key)
            T_FIRST[kind] += dt
        if STACK:
            NOW.inside[STACK[0]] += 1
        return out

    cls.__init__, cls.S = init, S

    for name in SOLVERS:
        def make(name, orig):
            def solver(self, *args, **kwargs):
                outermost = not STACK
                if outermost:
                    NOW.inversions[name] += 1
                STACK.append(name)
                try:
                    return orig(self, *args, **kwargs)
                except Exception:
                    if outermost:
                        NOW.failed[name] += 1
                    raise
                finally:
                    STACK.pop()
            return solver
        setattr(cls, name, make(name, getattr(cls, name)))


def run_script(work, name, args, log):
    """Run one script of the repository copy in `work` as if from the command line; its output goes to
    the string buffer `log`.  Returns the wall time in s."""
    path = os.path.join(work, name)
    argv, cwd = sys.argv, os.getcwd()
    sys.argv = [path] + list(args)
    os.chdir(work)
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            runpy.run_path(path, run_name='__main__')
    finally:
        wall = time.perf_counter() - t0
        sys.argv = argv
        os.chdir(cwd)
    return wall


# ---- 2. one capacitance, real and complex ---------------------------------------------------------
def biquadratic_S_complex(cell, eps_film, eps_sub):
    """fem_tensor.QuarterCell.S(eps_film, eps_sub, 'primal') in complex arithmetic: the same per-region
    matrices, the same reduction to the free nodes, the same SuperLU factorization (fem_tensor.solve_spd)
    and the same energy sum (no complex conjugates: S is continued analytically, as in ide_model.py and
    fem2d.py).  QuarterCell.S itself stores the potential as real numbers."""
    import fem_tensor as ft
    c = {'sub': eps_sub, 'film': eps_film, 'air': 1.0}
    K = sum(c[r] * cell.K[r] for r in cell.K)
    A, rhs = ft.reduce_to_free(K, cell.fixed_V, cell.V_bc)
    u = cell.V_bc.astype(complex)
    u[~cell.fixed_V], _ = ft.solve_spd(A, rhs)
    coef = sum(c[r] * cell.region[r] for r in cell.region)
    return 4.0 / np.pi * cell.mesh.energy(u, coef)


def one_S_times(im, fem2d):
    """Setup and each further S (real and complex) on the benchmark problem for the three methods.
    Returns a list of dicts.  Medians of repeats, as in the benchmark scripts."""
    import fem_tensor as ft
    import bench_fem_quadratic as bfq
    g = im.IDEGeometry(s_um=bc.S_GAP, w_um=bc.W_FINGER, te_um=bc.T_E)
    E, Q, Ec, Qc = bc.EPS_FILM, bc.EPS_SUB, EPS_FILM_CX, EPS_SUB_CX
    rows = []

    # the model (bench_model.py: setup median of 21, S median of 101)
    model, t_setup = bc.median_time(lambda: im.IDEModel(g), 21)
    model.S(E, Q, bc.T_P)                                           # (stores any modes it needs)
    S_r, t_r = bc.median_time(lambda: model.S(E, Q, bc.T_P), 101)
    S_c, t_c = bc.median_time(lambda: model.S(Ec, Qc, bc.T_P), 101)
    S_0 = model.S(E + 0j, Q + 0j, bc.T_P)
    rows.append(dict(method='model (ide_model.py), default settings', nodes=len(model.x), setup=t_setup,
                     real=t_r, complex=t_c, S=S_r, S_cx=S_c, check=abs(S_0 / S_r - 1)))

    # the biquadratic FEM (bench_fem_quadratic.py: medians of 7 or 21)
    params = dict(bfq.RUNGS)[BIQ_LABEL]
    cell, t_setup = bc.median_time(lambda: ft.QuarterCell(**params, **bfq.GEOM), 7)
    S_r, t_r = bc.median_time(lambda: cell.S(E, Q, 'primal'), 21)
    S_c, t_c = bc.median_time(lambda: biquadratic_S_complex(cell, Ec, Qc), 21)
    S_0 = biquadratic_S_complex(cell, E + 0j, Q + 0j)
    rows.append(dict(method=f'FEM biquadratic, {BIQ_LABEL}', nodes=cell.n_nodes, setup=t_setup,
                     real=t_r, complex=t_c, S=S_r, S_cx=S_c, check=abs(S_0 / S_r - 1)))
    del cell

    # fem2d.py (bench_fem2d.py: setup once, solve median of 3; here setup median of 3 as well)
    fem, t_setup = bc.median_time(lambda: fem2d.FEM2D(g, bc.T_P, **FEM2D_RUNG), 3)
    S_r, t_r = bc.median_time(lambda: fem.S(E, Q), 3)
    S_c, t_c = bc.median_time(lambda: fem.S(Ec, Qc), 3)
    S_0 = fem.S(E + 0j, Q + 0j)
    rows.append(dict(method='fem2d.py, h_c = t_e/32, growth 0.06', nodes=fem.n_nodes, setup=t_setup,
                     real=t_r, complex=t_c, S=S_r, S_cx=S_c, check=abs(S_0 / S_r - 1)))
    return rows


# ---- 3. the report --------------------------------------------------------------------------------
def fmt_time(t):
    """Seconds as a short readable string."""
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
    if bc.REPO is None:
        sys.exit('ide_model.py not found: set the environment variable DIELECTRIC_IDES to the repository folder')
    m = bc.machine_info()
    lines = [f"whole_analysis_timing.py  {time.strftime('%Y-%m-%d %H:%M')}",
             f"{m['cpu']} | {m['os']} | Python {m['python']}, numpy {m['numpy']}, scipy {m['scipy']} | "
             f"threads {m['threads']}",
             f'repository: {bc.REPO} (run on a temporary copy)', '']

    def say(text=''):
        print(text, flush=True)
        lines.append(text)

    print('\n'.join(lines), flush=True)

    # a temporary copy of the repository: the scripts write into its results\ folder, not the clone's
    tmp = tempfile.mkdtemp(prefix='whole_analysis_')
    work = os.path.join(tmp, 'dielectric-IDEs')
    shutil.copytree(bc.REPO, work, ignore=shutil.ignore_patterns('.git', '__pycache__', 'results', 'docs', 'benchmarks'))
    sys.path.insert(0, work)
    import ide_model as im
    import fem2d

    # 2. one capacitance with each method (before the counting wrappers are installed)
    print('timing one capacitance with each method ...', flush=True)
    rows = one_S_times(im, fem2d)

    # 1. the three scripts, counted
    global NOW
    instrument(im)
    counts, log = [], io.StringIO()
    for name, args in SCRIPTS:
        print(f'running {name} {" ".join(args)} ...', flush=True)
        NOW = Count(name, args)
        log.write(f'\n===== {name} {" ".join(args)}\n')
        NOW.wall = run_script(work, name, args, log)
        counts.append(NOW)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'w', encoding='utf-8') as fh:
        fh.write(log.getvalue())
    agree = [l.strip() for l in log.getvalue().splitlines() if 'values agree' in l]

    # ---- counts
    say('1. WHAT THE MODEL DOES IN THE PAPER\'S ANALYSIS')
    say(f"{'script':32s} {'wall s':>8s} {'in S s':>8s} {'setups':>7s} {'real S':>9s} {'complex S':>10s} "
        f"{'distinct S':>11s} {'geometries':>11s}")
    for c in counts:
        say(f"{c.label:32s} {c.wall:8.2f} {c.t['real'] + c.t['complex']:8.2f} {c.setups:7d} {c.n['real']:9,d} "
            f"{c.n['complex']:10,d} {len(c.distinct):11,d} {len(c.geometries):11d}")
    all_geom = set().union(*[c.geometries for c in counts])
    all_distinct = set().union(*[c.distinct for c in counts])
    tot = dict(wall=sum(c.wall for c in counts), inS=sum(c.t['real'] + c.t['complex'] for c in counts),
               setups=sum(c.setups for c in counts), t_setup=sum(c.t_setup for c in counts),
               real=sum(c.n['real'] for c in counts), complex=sum(c.n['complex'] for c in counts))
    d_real = sum(1 for d in all_distinct if d[0] == 'real')
    d_cx = len(all_distinct) - d_real
    say(f"{'all three':32s} {tot['wall']:8.2f} {tot['inS']:8.2f} {tot['setups']:7d} {tot['real']:9,d} "
        f"{tot['complex']:10,d} {len(all_distinct):11,d} {len(all_geom):11d}")
    say(f"   distinct evaluations: {d_real:,d} real, {d_cx:,d} complex.  Time in model setups: "
        f"{tot['t_setup']:.3f} s.  {'; '.join(agree)}")
    physical = {(s, w, te, round(t, 3)) for s, w, te, t in all_geom}
    blanks = {g for g in all_geom if g[3] <= g[2] + 1e-9}
    say(f'   geometries: {len(all_geom)} distinct (electrodes, film thickness) pairs, {len(blanks)} of them blanks; '
        f'{len(physical)} if thicknesses within 1 nm are merged (uncertainty.py evaluates S at t +- 0.1 nm '
        f'for dS/dt)')
    say('   root solves (each calls S many times):')
    for k in SOLVERS:
        n = sum(c.inversions[k] for c in counts)
        inside = sum(c.inside[k] for c in counts)
        failed = sum(c.failed[k] for c in counts)
        if n:
            say(f'     {k:24s} {n:6,d} calls, {inside / n:5.1f} evaluations each' +
                (f', {failed} failed (no solution)' if failed else ''))
    direct = tot['real'] + tot['complex'] - sum(sum(c.inside.values()) for c in counts)
    say(f'     outside the solvers (finite-difference derivatives of the uncertainty budget): {direct:,d} evaluations')
    say()

    # ---- one capacitance
    say('2. ONE CAPACITANCE ON THE BENCHMARK PROBLEM (setup reused; medians of repeats)')
    say(f"{'method':48s} {'nodes':>7s} {'setup':>9s} {'real S':>9s} {'complex S':>10s} {'cx/real':>8s}  "
        f"S (eps {bc.EPS_FILM:g}, {bc.EPS_SUB:g})   S (eps {EPS_FILM_CX:.4g}, {EPS_SUB_CX:.4g})")
    for r in rows:
        say(f"{r['method']:48s} {r['nodes']:7,d} {fmt_time(r['setup']):>9s} {fmt_time(r['real']):>9s} "
            f"{fmt_time(r['complex']):>10s} {r['complex'] / r['real']:8.2f}  {r['S']:.9f}   "
            f"{r['S_cx'].real:.9f}{r['S_cx'].imag:+.9f}j")
    say('   complex arithmetic with zero imaginary parts reproduces the real S to '
        + ', '.join(f"{r['check']:.0e}" for r in rows) + ' (same order)')
    say()

    # ---- estimates
    say('3. THE WHOLE ANALYSIS (all three scripts)')
    model = rows[0]
    est_model = tot['setups'] * model['setup'] + tot['real'] * model['real'] + tot['complex'] * model['complex']
    t_real = sum(c.t['real'] for c in counts)
    t_cx = sum(c.t['complex'] for c in counts)
    say(f"   model, measured: {fmt_time(tot['wall'])} wall time, of which {fmt_time(tot['inS'] + tot['t_setup'])} "
        f"inside the model (setups and S)")
    say(f"   the model's mean time per evaluation in the analysis: {fmt_time(t_real / max(tot['real'], 1))} real, "
        f"{fmt_time(t_cx / max(tot['complex'], 1))} complex (benchmark film: {fmt_time(model['real'])}, "
        f"{fmt_time(model['complex'])}).")
    say(f"   model, distinct evaluations only (each timed at its first occurrence): "
        f"{fmt_time(T_FIRST['real'] + T_FIRST['complex'])} inside S")
    say(f"   (For the model, blanks and thicker films are cheaper than the benchmark film, because fewer Fourier")
    say(f"   modes feel the film top; costed at the benchmark film's times, the counts give {fmt_time(est_model)}.")
    say(f"   An FEM solve costs about the same for every film, so the FEM estimates below use the benchmark times.)")
    say(f"   {'method':48s} {'as coded':>10s} {'distinct evaluations only':>27s}")
    for r in rows[1:]:
        full = len(all_geom) * r['setup'] + tot['real'] * r['real'] + tot['complex'] * r['complex']
        dist = len(all_geom) * r['setup'] + d_real * r['real'] + d_cx * r['complex']
        say(f"   {r['method']:48s} {fmt_time(full):>10s} {fmt_time(dist):>27s}")
    say(f'   (FEM: one setup per distinct geometry ({len(all_geom)}), one solve per evaluation of the model.)')
    say('   These replace every evaluation of the model by one solve.  A program written around a finite-element')
    say('   solver could need fewer solves (e.g. derivatives from the field of the same solve), so the FEM numbers')
    say('   are upper ends.  Relaxation (SOR) gets no estimate: its time per S depends on the starting potential.')

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'\nwritten to {OUT} (the scripts\' own output: {LOG})')


if __name__ == '__main__':
    main()
