"""
bench_sor_graded.py  (b1) -- the 5-point equations on graded grids, solved by red-black SOR.

Method 'five-point, SOR (graded grid)'.  Classic relaxation on the same grids as
bench_direct_graded.py (b2): each rung starts from V = 0 (electrode at V = 1) and runs until
S <= S_MODEL (time to target), or, on grids whose own S is above S_MODEL, until converged.  Because
S is an upper bound at every sweep, "S <= S_MODEL" means "at least as accurate as the model".

    python bench_sor_graded.py                    (about 3 min here; under 1 min on our desktop PC)
    python bench_sor_graded.py --rungs 3 4
    python bench_sor_graded.py --no-check         (leave out the convergence check)

The SOR (fivepoint.py): red-black ordering, compiled with Numba (compile time excluded), S checked
every 25 sweeps (about 1% of the time, included).  omega: Young's optimum from the EXACT Jacobi
spectral radius of each grid (scipy eigsh, shift-invert), then 2 - omega = 1.15 (2 - omega_opt),
the fastest fixed omega for reaching the target in scans (5% fewer sweeps than Young's here).  The
eigsh time is NOT included in solve_s (it costs about as much as a direct solve): generous to SOR.
"Converged" means the remaining decrease of S, extrapolated geometrically, is below 1e-7 S.

Expected output (Claude's sandbox, Xeon 2.1 GHz shared, 21 Nov 2026; sweep counts are
deterministic, times are noisy; solve_s = SOR time to the target, or to convergence for the rungs
whose own S is above the target):
    refine  nodes        1 - rho_J   omega         sweeps   S at stop   result           solve_s
    1          19,475    9.68e-06    1.989924659    1,675   7.1938966   converged, above   0.13 s
    1.5        43,452    4.80e-06    1.992893599    2,375   7.1906846   converged, above   0.36 s
    2          77,520    2.71e-06    1.994653105    3,150   7.1895386   converged, above   0.83 s
    2.5       120,870    1.65e-06    1.995827315    4,000   7.1890254   converged, above   1.7 s
    3         173,808    1.19e-06    1.996457385    3,225   7.1889051   TARGET reached     1.9 s
    4         307,692    7.08e-07    1.997266790    3,900   7.1889186   target reached     3.9 s
    5         480,496    4.45e-07    1.997832676    4,900   7.1888790   target reached     8.0 s
    6         691,740    3.15e-07    1.998174548    5,775   7.1888845   target reached    14.2 s
    8       1,229,256    1.78e-07    1.998629598    7,700   7.1888908   target reached    31.4 s
(3.0-4.0 ns per node-update including the S checks.)  The converged S of rungs 1-2.5 equal the
direct solutions of b2 to within 1e-7.  The fastest way to the target is the first rung that
reaches it (refine 3); finer grids need more sweeps on more nodes.
Run time: about 2 minutes here (eigsh included); about 30 s of SOR on our desktop PC.

CONVERGENCE CHECK.  After the ladder, the rung with the fewest nodes that reached the target is
followed by SOR on the grid with every spacing halved (refine x 2: refine 3 -> refine 6), started from
V = 0 and run to CONVERGENCE (remaining decrease below fivepoint.TOL_REL = 1e-7 of S), because a user
checking a grid does not know when to stop; omega from eigsh as for the rungs (not timed).  Logged as
'check: <rung label>' (bench_common.record_check), with the S-versus-time trajectory left out of
sor_trajectories.csv.  --no-check leaves it out (run_overnight.py runs single rungs with it).
Note that the rungs themselves stop as soon as S <= S_MODEL, which a user without the target could not
do either: generous to SOR.  Claude's sandbox (shared): refine 6 to convergence in 9,125 sweeps,
23.8 s (eigsh 7.3 s, not counted), S = 7.1882712 (direct solve 7.1882706); estimate +8.82e-05 (true
+1.12e-04); first S + check 25.7 s against 1.95 s for refine 3 (x13).
"""
import bench_common as bc      # first: one thread for every numerical library

import argparse
import time

import fivepoint as fp
from bench_direct_graded import LADDER, rung_label

METHOD = 'five-point, SOR (graded grid)'
# Sweeps to the target or to convergence (measured, deterministic); used by run_overnight.py.
EXPECTED_SWEEPS = {1: 1675, 1.5: 2375, 2: 3150, 2.5: 4000, 3: 3225, 4: 3900, 5: 4900, 6: 5775, 8: 7700}


def n_nodes(refine):
    xs, ys = fp.ladder_grid(refine)
    return xs.size * ys.size


def expected_sweeps(refine):
    return EXPECTED_SWEEPS.get(refine, round(1000 * refine))


def run_check(base_refine, base_S, limit):
    """SOR to convergence on the grid of refine 2 x base_refine (every spacing halved), from V = 0;
    logged with bench_common.record_check.  omega from eigsh, not timed (as for the rungs)."""
    refine = 2 * base_refine
    t0 = time.perf_counter()
    xs, ys = fp.ladder_grid(refine)
    wx, wy, fixed, V = fp.setup(xs, ys)
    setup_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    lam = fp.jacobi_lambda_exact(wx, wy, fixed)
    t_eig = time.perf_counter() - t1
    omega = fp.omega_sor(lam)
    label = bc.check_label(rung_label(base_refine))
    print(f'   {label}: refine {refine:g}, {V.size:,} nodes, 1 - rho_J = {lam:.4e}, omega = {omega:.9f}; '
          f'SOR to convergence', flush=True)
    res = fp.solve_sor(V, wx, wy, fixed, omega, target=None, time_limit_s=limit, method=METHOD, label=label,
                       log=False)
    ns = 1e9 * res['time_s'] / (res['sweeps'] * V.size)
    bc.record_check(METHOD, rung_label(base_refine), base_S, V.size, setup_s, res['time_s'], res['S'],
                    what=f'all element sizes / 2 (refine {refine:g})',
                    extra_notes=f"SOR from V = 0 to {res['reason']} (remaining < {fp.TOL_REL:g} S), {res['sweeps']} sweeps, "
                                f"{ns:.2f} ns/node-update; omega {omega:.9f} from eigsh ({t_eig:.1f} s, not counted)")


def main():
    p = argparse.ArgumentParser(description='b1: 5-point equations on graded grids, red-black SOR')
    p.add_argument('--rungs', type=float, nargs='+', default=LADDER, help='refine values (default: the full ladder)')
    p.add_argument('--max-minutes', type=float, default=None, help='time limit per rung (default none)')
    p.add_argument('--budget-hours', type=float, default=None, help='time limit for each rung = this (run_overnight.py)')
    p.add_argument('--no-check', action='store_true', help='leave out the convergence check (refine x 2)')
    a = p.parse_args()
    bc.banner('b1: bench_sor_graded.py -- five-point equations, graded grids (growth 0.12/refine), red-black SOR')
    fp.warm_up()
    limit = None if a.max_minutes is None else 60 * a.max_minutes
    if a.budget_hours is not None:
        limit = 3600 * a.budget_hours if limit is None else min(limit, 3600 * a.budget_hours)
    done = []
    for refine in a.rungs:
        res = fp.run_sor_rung(METHOD, rung_label(refine), lambda: fp.ladder_grid(refine), exact_lambda=True,
                              time_limit_s=limit)
        if res['hit']:                       # reached S <= S_MODEL: S at the first check below the target
            done.append(dict(refine=refine, n=res['n_nodes'], S=res['hit'][2]))
    if a.no_check:
        return
    base = bc.fewest_nodes(done, lambda r: bc.as_accurate_as_model(r['S']))
    if base is None:
        print('No rung reached the model accuracy: no convergence check.')
        return
    run_check(base['refine'], base['S'], limit)


if __name__ == '__main__':
    main()
