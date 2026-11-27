"""
experiments_fivepoint.py -- the experiments behind the choices made in fivepoint.py and the SOR
benchmarks, so that every number quoted in their docstrings can be checked.

    python experiments_fivepoint.py omega        (~2 min)  Jacobi rho_J: formula vs exact; omega scan
    python experiments_fivepoint.py families     (~1 min)  graded-grid families for b1/b2
    python experiments_fivepoint.py rectangles   (~1 min)  aspect ratios at equal node count (direct)
    python experiments_fivepoint.py uniform      (~2 min)  converged S of uniform grids versus h

Results in Claude's sandbox (21 Nov 2026) are quoted in each function's docstring.
"""
import bench_common as bc      # first: one thread for every numerical library

import math
import sys
import time

import numpy as np

import fivepoint as fp


def sweeps_to(V0, wx, wy, fixed, omega, S_inf, levels, every=5):
    """SOR from V0; sweeps (multiple of `every`) until (S - S_inf)/S_inf <= each level."""
    V, i_stop, k, out = V0.copy(), fp.free_ranges(fixed), 0, {}
    todo = sorted(levels, reverse=True)
    while todo:
        fp.sor_sweeps(V, wx, wy, i_stop, omega, every)
        k += every
        S = fp.energy_S(V, wx, wy)
        while todo and (S - S_inf) / S_inf <= todo[0]:
            out[todo.pop(0)] = k
    return out


def omega():
    """(1) 1 - rho_J from the constant-coefficient formula vs the exact value (eigsh).  Sandbox:
    exact/formula = 1.082, 1.086, 1.078 (squares h = 8, 4, 2 t_e), 1.083, 1.091 (16x4, 4x16 t_e),
    1.079, 1.080 (2:1 and 1:2 at 366k nodes), 1.081 (squares h = t_e).  Hence the factor 1.08.
    (2) omega scan on squares h = t_e (1.46M nodes): sweeps to a relative error of 7.7e-5, 4.5e-5,
    2.15e-5 (the margins of the target on finer grids) and 1e-7, for 2 - omega = c (2 - omega_opt):
      c = 1.00: 1240, 1280, 1335, 1735   c = 1.10: 1130, 1165, 1220, 1635   c = 1.15: 1090, 1125, 1180, 1705
      c = 1.20: 1060, 1100, 1160, 1815   c = 1.30: 1055, 1120, 1220, 2040   -> c = 1.15 used."""
    print('(1) 1 - rho_J: formula (constant coefficients) vs exact (eigsh)')
    for hx, hy in ((8, 8), (4, 4), (2, 2), (16, 4), (4, 16), (2 * 2 ** .5, 2 / 2 ** .5), (2 / 2 ** .5, 2 * 2 ** .5), (1, 1)):
        xs, ys = fp.uniform_grid(hx * bc.T_E, hy * bc.T_E)
        wx, wy, fixed, _ = fp.setup(xs, ys)
        lt, le = fp.jacobi_lambda_uniform(xs, ys), fp.jacobi_lambda_exact(wx, wy, fixed)
        print(f'   hx = {hx:5.3g} t_e, hy = {hy:5.3g} t_e: {xs.size * ys.size:9,} nodes  formula {lt:.4e}  exact {le:.4e}'
              f'  exact/formula {le / lt:.4f}', flush=True)
    print('(2) sweeps to relative error 7.7e-5, 4.5e-5, 2.15e-5, 1e-7 on squares h = t_e, 2 - omega = c (2 - omega_opt)')
    xs, ys = fp.uniform_grid(bc.T_E, bc.T_E)
    wx, wy, fixed, V0 = fp.setup(xs, ys)
    Vd, _, _ = fp.solve_direct(wx, wy, fixed, V0)
    S_inf = fp.energy_S(Vd, wx, wy)
    w_opt = fp.omega_opt(fp.lambda_uniform_calibrated(xs, ys))
    for c in (1.0, 1.1, 1.15, 1.2, 1.3):
        out = sweeps_to(V0, wx, wy, fixed, 2 - c * (2 - w_opt), S_inf, [7.7e-5, 4.5e-5, 2.15e-5, 1e-7])
        print(f'   c = {c:.2f}: ' + ', '.join(str(out[e]) for e in (7.7e-5, 4.5e-5, 2.15e-5, 1e-7)), flush=True)


def families():
    """Graded grids: S of the first rung with S <= S_MODEL in each family, and SOR sweeps to the
    target with Young's omega (exact rho_J).  Sandbox: growth 0.12 refine 6: 248,048 nodes,
    7.1888458, 4,050 sweeps; 0.12/sqrt(refine) refine 4: 180,360, 7.1888394, 3,925;
    0.12/refine refine 3: 173,808, 7.1887485, 3,400 (5% fewer with the factor 1.15).
    The rung below each misses: refine 5 (7.1890455), 3 (7.1892653), 2.5 (7.1890247)."""
    for name, refine, growth in (('growth 0.12', 6, 0.12), ('growth 0.12/sqrt(refine)', 4, 0.06),
                                 ('growth 0.12/refine', 3, 0.04)):
        xs, ys = fp.graded_grid(refine, growth)
        wx, wy, fixed, V0 = fp.setup(xs, ys)
        V, _, t = fp.solve_direct(wx, wy, fixed, V0)
        S = fp.energy_S(V, wx, wy)
        w = fp.omega_opt(fp.jacobi_lambda_exact(wx, wy, fixed))
        res = fp.solve_sor(V0.copy(), wx, wy, fixed, w, target=bc.S_MODEL, log=False, quiet=True)
        print(f'   {name:26s} refine {refine}: {V.size:,} nodes  S = {S:.7f}  direct {t:.2f} s  '
              f'SOR (Young omega) {res["sweeps"]} sweeps to the target', flush=True)


def rectangles():
    """Aspect ratio r = hx/hy at equal node count (cell area t_e^2, 1.46M nodes), direct solves.
    Sandbox: r = 1/4: 7.19796802, 1/2: 7.19292493, 1: 7.19204704, 2: 7.19264287, 4: 7.19422234.
    Converged SOR at 5.8M nodes (cell area (t_e/2)^2; too slow for this script): r = 1: 7.18969422,
    2: 7.18996595, 1/2: 7.19010298, 4: 7.19070799, 1/4: 7.19074762; sweeps to 1e-5: 2550, 2900,
    2800, 3825, 3625.  Squares best; 2:1 the best non-square ratio."""
    for r in (0.25, 0.5, 1.0, 2.0, 4.0):
        xs, ys = fp.uniform_grid(bc.T_E * math.sqrt(r), bc.T_E / math.sqrt(r))
        wx, wy, fixed, V0 = fp.setup(xs, ys)
        V, _, _ = fp.solve_direct(wx, wy, fixed, V0)
        print(f'   hx/hy = {r:4.2f}: {V.size:,} nodes  S = {fp.energy_S(V, wx, wy):.8f}  '
              f'1 - rho_J = {fp.lambda_uniform_calibrated(xs, ys):.3e}', flush=True)


def windowed_y(hy, D=3.0, growth=0.03, h_max=1.0):
    """y lines of uniform_grid(., hy) for |y| < D (to the next line beyond), graded outside."""
    ys = fp.uniform_lines(fp.Y_KEYS, hy)
    lo, hi = ys[ys <= -D].max(), ys[ys >= D].min()
    inner = ys[(ys >= lo) & (ys <= hi)]
    below = fp.graded_lines([-fp.Y_MAX, lo], [h_max, inner[1] - inner[0]], h_max, growth)
    above = fp.graded_lines([hi, fp.Y_MAX], [inner[-1] - inner[-2], h_max], h_max, growth)
    return np.concatenate([below[:-1], inner, above[1:]])


def uniform():
    """Converged S of uniform grids vs spacing.  Exact values: squares n = 1 (direct, below),
    n = 2, 3, 4 (SOR converged to 1e-10..1e-9; constants here): 7.19204704, 7.18969416,
    7.18909474, 7.18876452.  Finer grids are estimated on 'windowed' grids -- identical lines for
    |y| < 3 um, graded beyond -- which solve directly in seconds; their S is high by a bias that
    depends only on hy: 3.525e-5, 3.135e-5, 2.998e-5, 2.927e-5 for hy = t_e/1..4 (the 2:1
    rectangles give the same 3.135e-5, 2.998e-5 at hy = t_e/2, t_e/3), fitted as
    bias = B + b (t_e/hy)^-q and removed.  Sandbox: squares n = 5: 7.1885875; 2:1 rectangles
    m = 4..8: 7.1892915, 7.1889828, 7.1887911, 7.1886621, 7.1885749."""
    exact = {2: 7.1896941604, 3: 7.1890947424, 4: 7.1887645206}
    xs, ys = fp.uniform_grid(bc.T_E, bc.T_E)
    wx, wy, fixed, V0 = fp.setup(xs, ys)
    V, _, _ = fp.solve_direct(wx, wy, fixed, V0)
    exact[1] = fp.energy_S(V, wx, wy)
    print(f'   squares n = 1 (direct): S = {exact[1]:.8f}', flush=True)
    bias = {}
    for n in (1, 2, 3, 4):
        xs, ys = fp.uniform_lines(fp.X_KEYS, bc.T_E / n), windowed_y(bc.T_E / n)
        wx, wy, fixed, V0 = fp.setup(xs, ys)
        V, _, _ = fp.solve_direct(wx, wy, fixed, V0)
        bias[n] = fp.energy_S(V, wx, wy) / exact[n] - 1
        print(f'   squares n = {n}: windowed grid {V.size:,} nodes, bias {bias[n]:.4e}', flush=True)
    # fit bias = B + b n^-q (q by a 1D scan, B and b by least squares)
    nn, bb = np.array(list(bias)), np.array(list(bias.values()))
    best = min(((np.linalg.lstsq(np.c_[np.ones(4), nn ** -q], bb, rcond=None)[1].sum(), q)
                for q in np.arange(0.3, 3.0, 0.01)))
    q = best[1]
    (B, b), *_ = np.linalg.lstsq(np.c_[np.ones(4), nn ** -q], bb, rcond=None)
    print(f'   bias fit: {B:.4e} + {b:.4e} n^-{q:.2f}')
    for name, ratio, rungs in (('squares', 1.0, (5,)), ('2:1 rectangles', 2.0, (4, 5, 6, 7, 8))):
        for m in rungs:
            xs, ys = fp.uniform_lines(fp.X_KEYS, ratio * bc.T_E / m), windowed_y(bc.T_E / m)
            wx, wy, fixed, V0 = fp.setup(xs, ys)
            V, _, _ = fp.solve_direct(wx, wy, fixed, V0)
            S = fp.energy_S(V, wx, wy) / (1 + B + b * m ** -q)
            print(f'   {name} hy = t_e/{m}: estimated converged S = {S:.7f} ({100 * bc.rel_err(S):+.4f}%), '
                  f"{'beats' if S <= bc.S_MODEL else 'above'} S_MODEL", flush=True)


if __name__ == '__main__':
    parts = dict(omega=omega, families=families, rectangles=rectangles, uniform=uniform)
    if len(sys.argv) != 2 or sys.argv[1] not in parts:
        print(__doc__)
        sys.exit(1)
    bc.banner(f'experiments_fivepoint.py {sys.argv[1]}')
    fp.warm_up()
    t0 = time.perf_counter()
    parts[sys.argv[1]]()
    print(f'({time.perf_counter() - t0:.0f} s)')
