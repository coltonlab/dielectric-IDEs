"""
bench_sor_squares.py  (c1) -- classic relaxation: red-black SOR on UNIFORM square grids.

Method 'SOR, uniform squares'.  The textbook way to solve the problem: one spacing h = t_e/n
everywhere (the spacing in each interval between material boundaries is adjusted by a few per cent
so that every interface lies exactly on a grid line; the actual spacings are printed).  Each rung
starts from V = 0 (electrode at V = 1) and runs until S <= S_MODEL (time to target) or, if the
grid's own S is above S_MODEL, until converged (remaining decrease of S < 1e-7 S).

    python bench_sor_squares.py                          (default rungs; about 1 h on our desktop PC)
    python bench_sor_squares.py --rungs 1 2 3 4          (the rungs up to the first that reaches the target)
    python bench_sor_squares.py --rungs 8 --budget-hours 1.5
    (run_overnight.py runs the long rungs; this script can also be used directly)

--budget-hours B: before each rung, its time is estimated as nodes x expected sweeps x ns per
node-update (from the previous rung, or for the first rung from bc.calibrate()); a rung that would
not fit in what is left of B is skipped, and a running rung is stopped when B runs out.

omega: Young's optimum for the 5-point operator, with 1 - rho_J = 1.08 x the constant-coefficient
value (fivepoint.jacobi_lambda_uniform; the factor 1.08 is the measured ratio exact/formula, 1.078
to 1.091 on 7 grids), then 2 - omega = 1.15 (2 - omega_opt), the fastest fixed omega for reaching
the target in scans (12-14% fewer sweeps than Young's own omega).  S is checked every 25 sweeps.

RESULTS (Claude's sandbox, 21 Nov 2026).  S_grid is the CONVERGED S of each grid (an upper bound on
the true S); "sweeps" is the deterministic count to the target (or to convergence).
  n   h (nm)   nodes          S_grid         error      sweeps         reaches S_MODEL?
  1   115.0      1,457,773   7.19204704    +0.0549%    1,725 (conv)   no      measured
  2    57.5      5,823,435   7.18969416    +0.0222%    3,425 (conv)   no      measured
  3    38.3     13,093,850   7.18909474    +0.0139%   ~5,200 (conv)   no      measured S
  4    28.75    23,272,851   7.18876452    +0.0093%    5,025          YES     measured (first rung)
  5    23.0     36,357,303   7.1885875     +0.0068%   ~6,075          yes     S from a windowed grid
  6    19.2     52,352,429   7.188478      +0.0053%   ~7,200          yes     S extrapolated (error ~ h^1.39)
  8    14.4     93,088,618   7.188353      +0.0035%   ~9,600          yes     extrapolated
 10    11.5    145,432,694   7.188285      +0.0026%  ~12,000          yes     extrapolated
 12     9.6    209,397,182   7.188243      +0.0020%  ~14,450          yes     extrapolated
 16     7.2    372,215,250   7.188195      +0.0013%  ~19,400          yes     extrapolated
Speed of the kernel: about 3.9 ns per node-update (+2% for the S checks) in this sandbox (2.1 GHz Xeon,
shared, memory-bandwidth bound at ~8 GB/s on these large grids), so rung 4 takes about 8 min here.
Memory: 25 bytes per node (rung 16: 9.3 GB).  Rungs 1-2: this script.  S_grid of rungs 2-4: SOR
converged to 1e-9 or better, run in checkpointed pieces (sandbox commands are limited to 5 min); the
5,025 sweeps of rung 4 come from a run with exactly this kernel, omega rule and check interval.
The experiments behind these numbers and choices: experiments_fivepoint.py.
"""
import bench_common as bc      # first: one thread for every numerical library

import argparse
import time

import fivepoint as fp

METHOD = 'SOR, uniform squares'
RUNGS = (1, 2, 3, 4, 5, 6, 8)                  # run_overnight.py adds 10, 12, 16 if they fit
# Sweeps from V = 0 to the target (or to convergence), with the omega rule above and checks every
# 25 sweeps.  Rungs 1, 2, 4 measured; the others predicted from sweeps/n being nearly constant at
# a given relative accuracy (within 3% for n = 2, 3, 4).  Used only to estimate run times.
EXPECTED_SWEEPS = {1: 1725, 2: 3425, 3: 5200, 4: 5025, 5: 6075, 6: 7200, 8: 9600, 10: 12000,
                   12: 14450, 16: 19400}
SANDBOX_NS_PER_UPDATE = 4.1      # large uniform grids incl. S checks, Claude's sandbox (bc.calibrate() = 0.95 s)
SANDBOX_CALIBRATE_S = 0.95


def grid(n):
    return fp.uniform_grid(bc.T_E / n, bc.T_E / n)


def rung_label(n):
    return f'h = t_e/{n:g} = {1000 * bc.T_E / n:.2f} nm'


def n_nodes(n):
    xs, ys = grid(n)
    return xs.size * ys.size


def expected_sweeps(n):
    return EXPECTED_SWEEPS.get(n, round(1200 * n))


def run_ladder(method, rungs, make_grid, label, nodes, sweeps, budget_hours=None, ns_guess=None):
    """Run the rungs in the given order with fivepoint.run_sor_rung; with a budget, skip rungs whose
    estimated time does not fit in what is left, and stop a running rung when the budget runs out.
    ns_guess: ns per node-update for the first estimate (else from bc.calibrate())."""
    fp.warm_up()
    t_start = time.perf_counter()
    ns = ns_guess
    if budget_hours is not None and ns is None:
        cal = bc.calibrate()
        ns = SANDBOX_NS_PER_UPDATE * cal / SANDBOX_CALIBRATE_S
        print(f'bc.calibrate() = {cal:.2f} s (sandbox {SANDBOX_CALIBRATE_S} s): first guess {ns:.2f} ns per node-update', flush=True)
    for r in rungs:
        xs, ys = make_grid(r)
        print(f'\n{label(r)}\n{fp.spacing_report(xs, ys)}', flush=True)
        left = None if budget_hours is None else 3600 * budget_hours - (time.perf_counter() - t_start)
        if left is not None:
            est = nodes(r) * sweeps(r) * ns * 1e-9
            if est > left:
                print(f'   SKIPPED: estimated {est / 60:.0f} min ({nodes(r):,} nodes x {sweeps(r)} sweeps x '
                      f'{ns:.2f} ns) > {left / 60:.0f} min left in the budget', flush=True)
                continue
            print(f'   estimated {est / 60:.1f} min; {left / 60:.0f} min left in the budget', flush=True)
        res = fp.run_sor_rung(method, label(r), lambda: make_grid(r), exact_lambda=False, time_limit_s=left)
        ns = res['ns_per_update']


def main(method=METHOD, rungs=RUNGS):
    p = argparse.ArgumentParser(description='c1: red-black SOR on uniform square grids, h = t_e/n')
    p.add_argument('--rungs', type=float, nargs='+', default=rungs, help='values of n (h = t_e/n)')
    p.add_argument('--budget-hours', type=float, default=None, help='total time budget (default none)')
    p.add_argument('--ns-guess', type=float, default=None, help='ns per node-update for the first estimate (run_overnight.py)')
    a = p.parse_args()
    bc.banner('c1: bench_sor_squares.py -- red-black SOR on uniform squares h = t_e/n, from V = 0 to S <= S_MODEL')
    run_ladder(method, a.rungs, grid, rung_label, n_nodes, expected_sweeps, a.budget_hours, a.ns_guess)


if __name__ == '__main__':
    main()
