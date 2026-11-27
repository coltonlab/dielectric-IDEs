"""
bench_direct_graded.py  (b2) -- the 5-point equations on graded grids, solved directly with SuperLU.

Method 'five-point, direct (graded grid)'.  The same finite-difference equations and the same grids
as bench_sor_graded.py (b1), but solved with a sparse direct solver: no iteration error, so S is
the exact minimum of the stored energy on each grid (an upper bound on the true S).  The ladder runs
from coarse grids to beyond the point where S <= S_MODEL (the model's accuracy).

    python bench_direct_graded.py                    (about 1 min; 3 timing repeats per rung)
    python bench_direct_graded.py --rungs 3 4 --repeat 5
    python bench_direct_graded.py --no-check         (leave out the convergence check)

Times (median of --repeat runs; this sandbox is shared, so single timings are noisy):
    setup_s    grid, bond weights and assembly of the sparse matrix K_ff
    solve_s    SuperLU factorization (MMD_AT_PLUS_A ordering) + solve + S
    per_eval_s the solve time: each further S for new dielectric constants needs new weights and a
               new factorization

THE GRIDS (fivepoint.ladder_grid): relax_ide.py's grading -- spacing t_e/(8 refine) at the electrode
corners, at the substrate surface and at the electrode top, 4x that at the film top, growing by
`growth` times the distance up to 1/refine um (x) and 4/refine um (y) -- with growth = 0.12/refine,
so that the grading refines too.  Three families were compared (Claude's sandbox, 21 Nov 2026; S of
the first rung with S <= S_MODEL = 7.1889188, its nodes, direct time, and SOR sweeps to the target
with Young's omega from the exact rho_J):
    growth 0.12 fixed          refine 6: 248,048 nodes, S = 7.1888458, 0.90 s, 4,050 sweeps (1.0e9 updates)
    growth 0.12/sqrt(refine)   refine 4: 180,360 nodes, S = 7.1888394, 0.63 s, 3,925 sweeps (7.1e8)
    growth 0.12/refine         refine 3: 173,808 nodes, S = 7.1887485, 0.60 s, 3,400 sweeps (5.9e8)
growth 0.12/refine needs the least work for both the direct solve and SOR, so b1 and b2 use it.
(With growth 0.12 the ladder reproduces relax_ide.py: 7.193896, 7.191075, 7.189362, 7.188614 for
refine 1, 2, 4, 8.)

Expected output (Claude's sandbox: Xeon 2.1 GHz, shared; bc.calibrate() 0.95 s):
    refine  nodes       S           error      setup    solve
    1          19,475   7.1938961   +0.0807%   0.00 s   0.03 s
    1.5        43,452   7.1906841   +0.0360%   0.01 s   0.08 s
    2          77,520   7.1895381   +0.0200%   0.02 s   0.21 s
    2.5       120,870   7.1890247   +0.0129%   0.03 s   0.38 s
    3         173,808   7.1887485   +0.0091%   0.04 s   0.58 s   <- first rung with S <= S_MODEL
    4         307,692   7.1884749   +0.0052%   0.08 s   1.16 s
    5         480,496   7.1883470   +0.0035%   0.13 s   1.94 s
    6         691,740   7.1882706   +0.0024%   0.29 s   3.43 s
    8       1,229,256   7.1882015   +0.0014%   0.55 s   6.74 s
The ladder comes down toward S_REF from above and stays above the rigorous bracket's upper end
(7.1881700), as an upper bound must: the 5-point (linear-triangle) equations and fem2d.py's
bilinear elements converge to the same S.

CONVERGENCE CHECK.  After the ladder, the rung with the fewest nodes that beats the model is solved
once more on the grid with every spacing halved -- refine x 2, since the corner spacing, the caps and
the growth all scale as 1/refine (refine 3 -> refine 6: h_c = t_e/48, growth 0.02) -- and logged as
'check: <rung label>' (bench_common.record_check).  If that grid was already a rung of this run, its
timings are reused (same timing policy); otherwise it is run with the same --repeat.  --no-check
leaves it out.  Claude's sandbox (shared): refine 6, 691,740 nodes, S = 7.1882706; estimate +6.65e-05
(true +8.99e-05); setup + solve 4.3 s against 0.77 s for refine 3 (x6.7 for first S + check).
Run time: about 1 min with 3 repeats.
"""
import bench_common as bc      # first: one thread for every numerical library

import argparse
import time

import numpy as np
import scipy.sparse.linalg as spla

import fivepoint as fp

METHOD = 'five-point, direct (graded grid)'
LADDER = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)


def rung_label(refine):
    return f'refine {refine:g} (h_c = t_e/{8 * refine:g}, growth {0.12 / refine:.3g})'


def run_rung(refine):
    """Setup (grid, weights, matrix) and solve (factorize, solve, S), timed separately."""
    t0 = time.perf_counter()
    xs, ys = fp.ladder_grid(refine)
    wx, wy, fixed, V = fp.setup(xs, ys)
    Kff, rhs, free = fp.free_matrix(wx, wy, fixed, V)
    t1 = time.perf_counter()
    V.ravel()[free] = spla.splu(Kff, permc_spec='MMD_AT_PLUS_A').solve(rhs)
    S = fp.energy_S(V, wx, wy)
    return V.size, S, t1 - t0, time.perf_counter() - t1


def timed_rung(refine, repeat):
    """run_rung `repeat` times; returns (n, S, median setup s, median solve s)."""
    setups, solves = [], []
    for _ in range(repeat):
        n, S, ts, tv = run_rung(refine)
        setups.append(ts)
        solves.append(tv)
    return n, S, float(np.median(setups)), float(np.median(solves))   # median, as in the other scripts


def main():
    p = argparse.ArgumentParser(description='b2: 5-point equations on graded grids, direct solve')
    p.add_argument('--rungs', type=float, nargs='+', default=LADDER, help='refine values (default: the full ladder)')
    p.add_argument('--repeat', type=int, default=3, help='timing repeats per rung; the median is recorded')
    p.add_argument('--no-check', action='store_true', help='leave out the convergence check (refine x 2)')
    a = p.parse_args()
    bc.banner('b2: bench_direct_graded.py -- five-point equations, graded grids (growth 0.12/refine), SuperLU')
    fp.warm_up()
    done = {}
    for refine in a.rungs:
        n, S, setup_s, solve_s = timed_rung(refine, a.repeat)
        done[refine] = (n, S, setup_s, solve_s)
        bc.record(METHOD, rung_label(refine), n, setup_s, solve_s, S, per_eval_s=solve_s,
                  notes=f'median of {a.repeat} runs; SuperLU MMD_AT_PLUS_A')
    if a.no_check:
        return

    # convergence check of the cheapest rung that beats the model (see the module docstring)
    base = bc.fewest_nodes([dict(refine=r, n=v[0], S=v[1]) for r, v in done.items()],
                           lambda r: bc.as_accurate_as_model(r['S']))
    if base is None:
        print('No rung reached the model accuracy: no convergence check.')
        return
    refine2 = 2 * base['refine']
    reused = refine2 in done
    n, S, setup_s, solve_s = done[refine2] if reused else timed_rung(refine2, a.repeat)
    bc.record_check(METHOD, rung_label(base['refine']), base['S'], n, setup_s, solve_s, S,
                    what=f'all element sizes / 2 (refine {refine2:g})',
                    extra_notes=f"median of {a.repeat} runs{' (the refine ' + f'{refine2:g}' + ' rung above)' if reused else ''}; "
                                'SuperLU MMD_AT_PLUS_A')


if __name__ == '__main__':
    main()
