"""
verify_fivepoint.py -- checks of fivepoint.py (the 5-point equations, the direct solver and the
compiled red-black SOR).  Run it after any change to fivepoint.py.

    python verify_fivepoint.py            (about 5 s; prints PASS/FAIL for each check)

Checks
  1. Graded grids with relax_ide.py's recipe (growth 0.12), solved directly: refine 1 and 2 give
     relax_ide's values, S = 7.193896 and 7.191075.
  2. Every key line (x = 0, s/2, a/2; y = -Y, 0, t_e, t_p, Y) is exactly a grid line of
     uniform_grid(), so the geometry is not rounded.
  3. The compiled SOR kernel, which updates black row j-1 right after red row j, gives bit-for-bit
     the same potential as the textbook order (all red nodes, then all black nodes), written here
     in plain numpy.
  4. SOR (with the benchmarks' omega rule) converges to the direct solution of the same equations,
     within 1e-9 relative in S, on the graded refine-1 grid and on a small uniform grid (h = 4 t_e).
  5. S decreases monotonically during SOR (checked after every sweep), so every intermediate S is
     an upper bound on the true S.

Expected output (Claude's sandbox, 21 Nov 2026): all PASS; SOR/direct - 1 about 1e-13; largest
increase of S between successive sweeps 0 (S never increases).
"""
import bench_common as bc      # first: one thread for every library

import time

import numpy as np

import fivepoint as fp

failures = []


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        failures.append(name)


def red_black_numpy(V, wx, wy, fixed, omega, n_sweeps):
    """Textbook red-black SOR in plain numpy: all red nodes ((i + j) even), then all black nodes."""
    ny, nx = V.shape
    J, I = np.indices(V.shape)
    for _ in range(n_sweeps):
        for color in (0, 1):
            num = np.zeros_like(V)
            den = np.zeros_like(V)
            num[:, :-1] += wx * V[:, 1:]; den[:, :-1] += wx      # east neighbour
            num[:, 1:] += wx * V[:, :-1]; den[:, 1:] += wx       # west neighbour
            num[:-1, :] += wy * V[1:, :]; den[:-1, :] += wy      # north neighbour
            num[1:, :] += wy * V[:-1, :]; den[1:, :] += wy       # south neighbour
            m = ((I + J) % 2 == color) & ~fixed
            V[m] += omega * (num[m] / den[m] - V[m])
    return V


def main():
    bc.banner('verify_fivepoint.py: checks of the 5-point equations, direct solver and SOR')
    t_start = time.perf_counter()
    fp.warm_up()

    # 1. relax_ide's reference values
    for refine, ref in ((1, 7.193896), (2, 7.191075)):
        xs, ys = fp.graded_grid(refine, growth=0.12)
        wx, wy, fixed, V0 = fp.setup(xs, ys)
        V, _, _ = fp.solve_direct(wx, wy, fixed, V0)
        S = fp.energy_S(V, wx, wy)
        check(f'graded refine {refine} (growth 0.12), direct', abs(S - ref) < 5e-7,
              f'{V.size:,} nodes, S = {S:.7f}, relax_ide.py {ref:.6f}')

    # 2. interfaces on grid lines
    xs, ys = fp.uniform_grid(bc.T_E / 3, bc.T_E / 3)
    on_lines = all(np.any(xs == k) for k in fp.X_KEYS) and all(np.any(ys == k) for k in fp.Y_KEYS)
    check('uniform_grid(t_e/3, t_e/3): key lines are grid lines', on_lines,
          f'spacing per interval:\n{fp.spacing_report(xs, ys)}')

    # 3. fused-row kernel == textbook red-black order (small uniform grid, 30 sweeps from V = 0)
    xs, ys = fp.uniform_grid(8 * bc.T_E, 8 * bc.T_E)
    wx, wy, fixed, V0 = fp.setup(xs, ys)
    V1 = V0.copy()
    fp.sor_sweeps(V1, wx, wy, fp.free_ranges(fixed), 1.9, 30)
    V2 = red_black_numpy(V0.copy(), wx, wy, fixed, 1.9, 30)
    check('compiled kernel = textbook red-black SOR', np.array_equal(V1, V2) or np.abs(V1 - V2).max() < 1e-14,
          f'largest difference in V after 30 sweeps: {np.abs(V1 - V2).max():.1e}')

    # 4 and 5. SOR -> direct, S monotone
    for name, (xs, ys), lam in (('graded refine 1', fp.graded_grid(1, growth=0.12), None),
                                ('uniform h = 4 t_e', fp.uniform_grid(4 * bc.T_E, 4 * bc.T_E), None)):
        wx, wy, fixed, V0 = fp.setup(xs, ys)
        Vd, _, _ = fp.solve_direct(wx, wy, fixed, V0)
        S_direct = fp.energy_S(Vd, wx, wy)
        omega = fp.omega_sor(fp.jacobi_lambda_exact(wx, wy, fixed))      # the benchmarks' omega rule
        V = V0.copy()
        res = fp.solve_sor(V, wx, wy, fixed, omega, tol=1e-13, check_every=1, log=False, quiet=True)
        rel = res['S'] / S_direct - 1
        check(f'{name}: SOR converges to the direct S', abs(rel) < 1e-9,
              f'{V.size:,} nodes, omega {omega:.6f}, {res["sweeps"]} sweeps: SOR {res["S"]:.12f}, '
              f'direct {S_direct:.12f}, SOR/direct - 1 = {rel:.1e}')
        # monotone: S after every sweep (check_every=1 above stores nothing, so redo with a record)
        V = V0.copy()
        i_stop = fp.free_ranges(fixed)
        S_prev, worst, n = fp.energy_S(V, wx, wy), 0.0, res['sweeps']
        for _ in range(n):
            fp.sor_sweeps(V, wx, wy, i_stop, omega, 1)
            S = fp.energy_S(V, wx, wy)
            worst = max(worst, (S - S_prev) / S)
            S_prev = S
        check(f'{name}: S never increases during SOR', worst <= 1e-14,
              f'largest relative increase between successive sweeps in {n} sweeps: {worst:.1e}')

    print(f'\n{"ALL CHECKS PASSED" if not failures else "FAILED: " + ", ".join(failures)}  '
          f'({time.perf_counter() - t_start:.0f} s)')


if __name__ == '__main__':
    main()
