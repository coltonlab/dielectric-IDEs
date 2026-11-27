"""
fivepoint.py -- five-point finite differences for the IDE quarter cell: grids, equations, stored
energy, a direct solver, and red-black SOR (successive over-relaxation) compiled with Numba.

Library for bench_direct_graded.py (b2), bench_sor_graded.py (b1), bench_sor_squares.py (c1) and
bench_sor_rectangles.py (c2).  It is also an independent check of fem2d.py: a different
discretization (linear triangles instead of bilinear rectangles) and a different solver must
converge to the same S.  Import bench_common BEFORE this module (it pins every library to one thread).

THE PROBLEM (same as fem2d.py and relax_ide.py)
-----------------------------------------------
One quarter of a period, 0 < x < a/2, -Y < y < Y with Y = 3a.  V = 0 at x = 0 (centre of a gap) and
at y = +-Y; the electrode (s/2 < x < a/2, 0 < y < t_e) is at V = 1; x = a/2 (centre of an electrode)
is a symmetry plane, dV/dx = 0.  Substrate (eps_sub) for y < 0, film (eps_film) in the gap up to
y = t_p and on top of the electrode up to y = t_p, air above.

THE EQUATIONS
-------------
Tensor-product grid: x lines xs[0..nx-1] from 0 to a/2, y lines ys[0..ny-1] from -Y to +Y.  Every
material boundary and electrode surface lies exactly on a grid line (moving the film top by 7 nm
changes S by 0.26%, so rounding the geometry to the grid is not an option).  Arrays are indexed
[j, i] = [y, x].  Each grid cell has one dielectric constant.  A "bond" between neighbouring nodes
gets a weight from the (up to two) cells that share it:
    x-bond (j,i)-(j,i+1):  wx[j,i] = sum over the cells below and above it of  eps * hy_cell / (2 hx)
    y-bond (j,i)-(j+1,i):  wy[j,i] = sum over the cells left and right of it of eps * hx_cell / (2 hy)
At every free node:  sum over its bonds of  w * (V_neighbour - V_node) = 0.
Nodes on x = a/2 have no bond to the east: that alone is the symmetry (Neumann) condition.

These are exactly the finite-element equations for LINEAR TRIANGLES (each cell cut in two along a
diagonal), and
    S = (4/pi) * sum over all bonds of  w * (Delta V)^2
is the stored energy of that piecewise-linear potential, in the normalization of fem2d.py and
ide_model.py (C = geom.prefactor() * S).  By Dirichlet's principle S is an UPPER bound on the true S
for ANY V that obeys the boundary conditions, converged or not.  The solution of the equations is the
V that minimizes S on the grid, and every SOR update with 0 < omega < 2 lowers S, so during SOR S
comes down monotonically and stays an upper bound all the way.

CONTENTS
--------
graded_grid(refine, growth)  grid graded toward the electrode corners (relax_ide.build_grid's recipe)
ladder_grid(refine)          the graded grids of the b1/b2 benchmarks (growth 0.12/refine)
uniform_grid(hx, hy)         uniform spacing, adjusted per interval so every interface is a grid line
setup(xs, ys)                bond weights wx, wy, fixed-node mask, initial V (1e-30, electrode 1)
energy_S(V, wx, wy)          S from the stored energy (compiled, one pass)
solve_direct(...)            SuperLU on the same equations
jacobi_lambda_exact(...)     1 - rho_J (Jacobi spectral radius) by eigsh; jacobi_lambda_uniform: formula
omega_opt, omega_sor         Young's optimal omega; the omega used in the benchmarks
sor_sweeps(...)              the compiled red-black SOR kernel (warm_up() compiles it before timing)
solve_sor(...)               SOR with S checks, stopping rules and a trajectory log
run_sor_rung(...)            one benchmark rung: grid, omega, SOR to the target, bc.record
"""
import csv
import datetime
import math
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit

import bench_common as bc

S_GAP, T_E, A, T_P = bc.S_GAP, bc.T_E, bc.A, bc.T_P       # gap s, t_e, half-period a, film top (um)
Y_MAX = bc.Y_EXTENT_IN_A * A                               # V = 0 at y = +-Y_MAX
X_KEYS = (0.0, S_GAP / 2, A / 2)                           # lines that every grid must contain
Y_KEYS = (-Y_MAX, 0.0, T_E, T_P, Y_MAX)
TRAJ_CSV = os.path.join(bc.RESULTS_DIR, 'sor_trajectories.csv')

# Default stopping rule for SOR runs that do not reach the target (see solve_sor): stop when the
# extrapolated remaining decrease of S is below TOL_REL * S.  1e-7 is under 0.1% of the model's own
# error (1.1e-4 relative), hence negligible against the discretization error of any grid whose
# converged S is above the target.
TOL_REL = 1e-7
CHECK_EVERY = 25     # sweeps between evaluations of S: overhead under 2% (an S costs 0.2-0.45 sweep)


# ================================================================================================
# Grids
# ================================================================================================
def graded_lines(keys, h_keys, h_max, growth):
    """Grid lines along one axis (same as relax_ide.graded_lines and fem2d.graded_axis).  `keys`
    are lines; near key k the spacing is h_keys[k], growing by `growth` times the distance from the
    nearest key, up to h_max."""
    pts = [keys[0]]
    for xa, xb, ha, hb in zip(keys[:-1], keys[1:], h_keys[:-1], h_keys[1:]):
        x, seg = xa, []
        while True:
            h = min(h_max, ha + growth * (x - xa), hb + growth * (xb - x))
            if x + 1.3 * h >= xb:
                break
            x += h
            seg.append(x)
        pts += seg + [xb]
    return np.array(pts)


def graded_grid(refine, growth=0.12):
    """relax_ide.build_grid's grid: spacing t_e/(8 refine) at the electrode corners and at the
    substrate surface and electrode top, 4x that at the film top, growing by `growth` times the
    distance, up to 1/refine um in x and 4/refine um in y.  Returns the x and y lines (um)."""
    hc = T_E / 8.0 / refine
    hx_max, hy_max = 1.0 / refine, 4.0 / refine
    xs = graded_lines(X_KEYS, [hx_max, hc, hx_max], hx_max, growth)
    ys = graded_lines(Y_KEYS, [hy_max, hc, hc, 4 * hc, hy_max], hy_max, growth)
    return xs, ys


def ladder_grid(refine):
    """The graded grids of the b1/b2 benchmarks: graded_grid with growth 0.12/refine, so that the
    grading also refines (as in the paper's FEM ladder).  Of the three families tried -- growth
    0.12 fixed, 0.12/sqrt(refine), 0.12/refine -- this one reaches S <= S_MODEL with the fewest
    nodes (174k at refine 3, against 180k at refine 4 and 248k at refine 6), the shortest direct
    solve (0.6 s against 0.63 s and 0.9 s here) and the fewest SOR node-updates (5.9e8 against
    7.1e8 and 1.0e9).  Data: notes in bench_direct_graded.py."""
    return graded_grid(refine, growth=0.12 / refine)


def uniform_lines(keys, h):
    """Each interval between consecutive keys is cut into n = max(1, round(L/h)) equal parts, so the
    spacing is within a few per cent of h and every key is exactly a grid line."""
    pts = [keys[0]]
    for xa, xb in zip(keys[:-1], keys[1:]):
        n = max(1, int(round((xb - xa) / h)))
        pts.extend(np.linspace(xa, xb, n + 1)[1:])
    return np.array(pts)


def uniform_grid(hx, hy):
    """Uniform grid with spacing about hx in x and hy in y (squares if hx = hy)."""
    return uniform_lines(X_KEYS, hx), uniform_lines(Y_KEYS, hy)


def spacing_report(xs, ys):
    """One line per key interval: number of parts and actual spacing (nm)."""
    out = []
    for name, lines, keys in (('x', xs, X_KEYS), ('y', ys, Y_KEYS)):
        for xa, xb in zip(keys[:-1], keys[1:]):
            seg = lines[(lines >= xa - 1e-9) & (lines <= xb + 1e-9)]
            h = np.diff(seg)
            out.append(f'{name} {xa:9.4f}..{xb:9.4f} um: {len(h):7d} parts, spacing '
                       f'{1000 * h.min():.3f}' + ('' if h.max() - h.min() < 1e-9 else f'..{1000 * h.max():.3f}') + ' nm')
    return '\n'.join(out)


# ================================================================================================
# The equations: bond weights, fixed nodes, initial potential
# ================================================================================================
@njit(cache=True)
def _fill_bonds(xs, ys, xcls, ycls, eps_table, wx, wy):
    """Bond weights from the cell dielectric constants.  Cell (j, i) lies between lines j, j+1 and
    i, i+1; its eps is eps_table[ycls[j], xcls[i]].  Written as loops so that no full-size
    temporary array is needed (grids of 1e8+ nodes)."""
    nx, ny = xs.size, ys.size
    for j in range(ny):                       # x-bonds (j,i)-(j,i+1)
        for i in range(nx - 1):
            w = 0.0
            if j > 0:                         # cell below the bond
                w += eps_table[ycls[j - 1], xcls[i]] * (ys[j] - ys[j - 1])
            if j < ny - 1:                    # cell above the bond
                w += eps_table[ycls[j], xcls[i]] * (ys[j + 1] - ys[j])
            wx[j, i] = w / (2.0 * (xs[i + 1] - xs[i]))
    for j in range(ny - 1):                   # y-bonds (j,i)-(j+1,i)
        for i in range(nx):
            w = 0.0
            if i > 0:                         # cell to the left
                w += eps_table[ycls[j], xcls[i - 1]] * (xs[i] - xs[i - 1])
            if i < nx - 1:                    # cell to the right (none on x = a/2)
                w += eps_table[ycls[j], xcls[i]] * (xs[i + 1] - xs[i])
            wy[j, i] = w / (2.0 * (ys[j + 1] - ys[j]))


def _line_index(lines, value):
    """Index of the grid line at `value`; it must be there exactly (interfaces lie on grid lines)."""
    k = int(np.argmin(np.abs(lines - value)))
    if abs(lines[k] - value) > 1e-12 * max(1.0, abs(value)):
        raise ValueError(f'{value} um is not a grid line')
    return k


def setup(xs, ys, eps_film=bc.EPS_FILM, eps_sub=bc.EPS_SUB):
    """Bond weights wx (ny, nx-1) and wy (ny-1, nx), fixed-node mask (ny, nx) and the starting
    potential V (1 on the electrode, 0 on the outer boundaries, 1e-30 at the free nodes; see the
    comment below).  Memory: 25 bytes per node, no temporaries."""
    nx, ny = xs.size, ys.size
    # material of each cell from two 1D class arrays (cell centres never lie on a key line)
    xcls = (0.5 * (xs[1:] + xs[:-1]) > S_GAP / 2).astype(np.int64)    # 0 gap side, 1 electrode side
    ycls = np.searchsorted([0.0, T_E, T_P], 0.5 * (ys[1:] + ys[:-1])).astype(np.int64)
    #        ycls: 0 substrate (y < 0), 1 electrode layer (0 < y < t_e), 2 t_e < y < t_p, 3 air
    eps_table = np.array([[eps_sub, eps_sub],      # substrate
                          [eps_film, 1.0],          # film in the gap | inside the electrode (never matters:
                          [eps_film, eps_film],     # film            |  all four corners are at V = 1)
                          [1.0, 1.0]])              # air
    wx = np.empty((ny, nx - 1))
    wy = np.empty((ny - 1, nx))
    _fill_bonds(xs, ys, xcls, ycls, eps_table, wx, wy)

    i_edge, j_sub, j_top = _line_index(xs, S_GAP / 2), _line_index(ys, 0.0), _line_index(ys, T_E)
    fixed = np.zeros((ny, nx), dtype=np.bool_)
    fixed[:, 0] = True                              # x = 0: V = 0 (antisymmetry, centre of a gap)
    fixed[0, :] = True                              # y = -Y: V = 0
    fixed[-1, :] = True                             # y = +Y: V = 0
    fixed[j_sub:j_top + 1, i_edge:] = True          # electrode, surface included: V = 1
    # The free nodes start at 1e-30, not 0 (changed 22 Nov 2026).  From V = 0, SOR spreads a wave of
    # tiny values ahead of the solution, and those below 2.2e-308 are subnormal numbers, which the
    # processor handles many times more slowly: up to 4% of the nodes were subnormal, and the
    # uniform-grid runs slowed 4-5x for most of their time.  From 1e-30 every value stays normal; the
    # sweeps are the same and S changes only by rounding (about 1e-15 relative, h = t_e/1 and t_e/2).
    # solve_direct uses only the fixed nodes of V: no change there.
    V = np.full((ny, nx), 1e-30)
    V[:, 0] = 0.0                                   # the fixed nodes, as in the mask above
    V[0, :] = 0.0
    V[-1, :] = 0.0
    V[j_sub:j_top + 1, i_edge:] = 1.0
    return wx, wy, fixed, V


# ================================================================================================
# Stored energy
# ================================================================================================
@njit(cache=True, fastmath=True)
def energy_S(V, wx, wy):
    """S = (4/pi) sum_bonds w (Delta V)^2, in one pass over the arrays.  Summed row by row, then the
    rows added, which keeps the rounding error of the sum near 1e-14 relative even for 1e8 nodes.
    fastmath lets the compiler reorder (vectorize) the sum: 35% faster, S changes by ~1e-14."""
    ny, nx = V.shape
    total = 0.0
    for j in range(ny):
        row = 0.0
        for i in range(nx - 1):
            d = V[j, i + 1] - V[j, i]
            row += wx[j, i] * d * d
        if j < ny - 1:
            for i in range(nx):
                d = V[j + 1, i] - V[j, i]
                row += wy[j, i] * d * d
        total += row
    return 4.0 / math.pi * total


# ================================================================================================
# Direct solution of the same equations (SuperLU) -- also used for the exact Jacobi spectral radius
# ================================================================================================
def free_matrix(wx, wy, fixed, V):
    """Matrix K_ff of the equations at the free nodes, and right side b = -K_fc V_c (the fixed nodes
    moved to the right).  K = sum over bonds of w (e_a - e_b)(e_a - e_b)^T."""
    ny, nx = fixed.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])   # bond ends
    b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    w = np.concatenate([wx.ravel(), wy.ravel()])
    K = sp.csr_matrix((np.concatenate([w, w, -w, -w]),
                       (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))), shape=(nx * ny,) * 2)
    fix = fixed.ravel()
    free = ~fix
    Kf = K[free]
    return Kf[:, free].tocsc(), -(Kf[:, fix] @ V.ravel()[fix]), free


def solve_direct(wx, wy, fixed, V0, permc_spec='MMD_AT_PLUS_A'):
    """Solve the 5-point equations with SuperLU.  Returns V (new array) and the times (s) for
    building the matrix and for factorizing + solving.  MMD_AT_PLUS_A (minimum degree on K + K^T)
    is the fastest SuperLU ordering for these symmetric matrices (2-3x faster than the default
    COLAMD at 4e5 nodes)."""
    t0 = time.perf_counter()
    Kff, rhs, free = free_matrix(wx, wy, fixed, V0)
    t1 = time.perf_counter()
    V = V0.copy()
    V.ravel()[free] = spla.splu(Kff, permc_spec=permc_spec).solve(rhs)
    return V, t1 - t0, time.perf_counter() - t1


# ================================================================================================
# Spectral radius of the Jacobi iteration, and the optimal over-relaxation factor
# ================================================================================================
# Jacobi replaces each free node by the w-weighted average of its neighbours: V <- D^-1 (D - K) V,
# D = diag(K).  Its spectral radius is rho_J = 1 - lambda_min, where lambda_min is the smallest
# eigenvalue of  K v = lambda D v  (the spectrum of D^-1 K is symmetric about 1 because the grid is
# red-black bipartite).  Red-black ordering of the 5-point stencil is "consistently ordered", so
# Young's theory holds exactly: the best SOR factor is omega_opt = 2 / (1 + sqrt(1 - rho_J^2)), and
# then the error shrinks by about (omega_opt - 1) per sweep.

def omega_opt(lam_min):
    """Young's optimal omega from lambda_min = 1 - rho_J (written so that it keeps full precision
    when rho_J is within 1e-9 of 1:  1 - rho_J^2 = lambda (2 - lambda))."""
    return 2.0 / (1.0 + math.sqrt(lam_min * (2.0 - lam_min)))


def jacobi_lambda_exact(wx, wy, fixed):
    """lambda_min of K v = lambda D v on the free nodes, by shift-invert Lanczos (scipy eigsh,
    sigma = 0, which needs one sparse LU factorization of K: about the cost of a direct solve)."""
    Kff, _, _ = free_matrix(wx, wy, fixed, np.zeros(fixed.shape))
    D = sp.diags(Kff.diagonal()).tocsc()
    lu = spla.splu(Kff, permc_spec='MMD_AT_PLUS_A')
    OPinv = spla.LinearOperator(Kff.shape, matvec=lu.solve, dtype=float)
    lam = spla.eigsh(Kff, k=1, M=D, sigma=0.0, which='LM', OPinv=OPinv, tol=1e-10,
                     return_eigenvectors=False)
    return float(lam[0])


def jacobi_lambda_uniform(xs, ys):
    """Theory for (nearly) uniform grids: the constant-coefficient 5-point operator on this domain.
    Its slowest mode is sin(pi x / a) cos(pi y / 2Y) (V = 0 at x = 0, dV/dx = 0 at x = a/2,
    V = 0 at y = +-Y), with Jacobi eigenvalue
        rho_J = (wx cos(pi hx / a) + wy cos(pi hy / 2Y)) / (wx + wy),  wx = hy/hx, wy = hx/hy
    (wx, wy are the bond weights of an hx-by-hy cell, so rectangles are included).  The mean
    spacings are used; the electrode, the materials and the few-per-cent spacing changes between
    key intervals are ignored -- validated against jacobi_lambda_exact in the notes."""
    hx = (xs[-1] - xs[0]) / (xs.size - 1)
    hy = (ys[-1] - ys[0]) / (ys.size - 1)
    wx, wy = hy / hx, hx / hy
    one_minus_cos_x = 2.0 * math.sin(math.pi * hx / A / 2.0) ** 2      # 1 - cos, without cancellation
    one_minus_cos_y = 2.0 * math.sin(math.pi * hy / (2.0 * Y_MAX) / 2.0) ** 2
    return (wx * one_minus_cos_x + wy * one_minus_cos_y) / (wx + wy)


# The omega used in every SOR benchmark (SOR's best case; see the notes in the scripts):
#  * Uniform grids: lambda_min = 1.08 x the constant-coefficient value above.  The exact value
#    (eigsh, 7 uniform grids of 23k-1.46M nodes, squares and rectangles hx/hy = 1/4..4) is 1.078 to
#    1.091 times the formula: the electrode and the dielectrics speed up the slowest mode a little.
#  * Graded grids: lambda_min from eigsh (jacobi_lambda_exact), computed before the timed run.
#  * Young's omega_opt minimizes the ASYMPTOTIC error per sweep.  To reach a moderate accuracy from
#    V = 0, a slightly smaller omega is faster: scans give the fewest sweeps to the target for
#    2 - omega about 1.15-1.2 times Young's (uniform: 12-14% fewer sweeps than Young; graded: 5%;
#    still 2% better at 1e-7 convergence).  We use 2 - omega = 1.15 (2 - omega_opt) everywhere.
LAMBDA_CORRECTION = 1.08
OMEGA_FACTOR = 1.15


def omega_sor(lam_min):
    """omega for the SOR benchmarks: 2 - 1.15 (2 - omega_opt)."""
    return 2.0 - OMEGA_FACTOR * (2.0 - omega_opt(lam_min))


def lambda_uniform_calibrated(xs, ys):
    """lambda_min = 1 - rho_J for a uniform grid: 1.08 x the constant-coefficient formula."""
    return LAMBDA_CORRECTION * jacobi_lambda_uniform(xs, ys)


# ================================================================================================
# The SOR kernel
# ================================================================================================
def free_ranges(fixed):
    """For each row j, i_stop[j] such that the free nodes of the row are exactly i = 1 .. i_stop[j]-1.
    The SOR kernel uses these ranges instead of testing the mask at every node (15% faster).  True
    for this geometry: node 0 of every row is on x = 0; the rows 0 <= y <= t_e end in the electrode;
    the rows y = +-Y are all fixed.  Checked here against the mask."""
    ny, nx = fixed.shape
    after0 = fixed[:, 1:]
    i_stop = np.where(after0.any(axis=1), after0.argmax(axis=1) + 1, nx).astype(np.int64)
    if not (fixed[:, 0].all() and np.array_equal(fixed.sum(axis=1), 1 + nx - i_stop)):
        raise ValueError('fixed nodes are not "x = 0 plus the end of each row"')
    return i_stop


@njit(cache=True)
def _relax_row(V, wx, wy, i_stop, omega, j, color):
    """Over-relax the free nodes of row j whose colour (i + j) % 2 equals `color`.  Each one moves
    from V toward the Gauss-Seidel value (the w-weighted average of its 4 neighbours) by omega times
    the difference."""
    nx = V.shape[1]
    stop = i_stop[j]
    i0 = 1 if (1 + j) % 2 == color else 2       # i = 0 is the fixed line x = 0
    for i in range(i0, min(stop, nx - 1), 2):
        we, ww, wn, ws = wx[j, i], wx[j, i - 1], wy[j, i], wy[j - 1, i]
        avg = (we * V[j, i + 1] + ww * V[j, i - 1] + wn * V[j + 1, i] + ws * V[j - 1, i]) / (we + ww + wn + ws)
        V[j, i] += omega * (avg - V[j, i])
    i = nx - 1                                   # last node, on the symmetry plane x = a/2, if free:
    if stop == nx and (i + j) % 2 == color:      # no east neighbour
        ww, wn, ws = wx[j, i - 1], wy[j, i], wy[j - 1, i]
        avg = (ww * V[j, i - 1] + wn * V[j + 1, i] + ws * V[j - 1, i]) / (ww + wn + ws)
        V[j, i] += omega * (avg - V[j, i])


@njit(cache=True)
def sor_sweeps(V, wx, wy, i_stop, omega, n_sweeps):
    """n_sweeps red-black SOR sweeps, in place.  Rows 0 and ny-1 (y = -Y, +Y) are fixed.
    Red-black: first every node with (i + j) even ("red"), then every node with (i + j) odd
    ("black").  A red node's neighbours are all black and vice versa, so the order within a colour
    does not matter.  Here the black update runs one row behind the red one (red row j, then black
    row j-1): when red row j is done, black row j-1 has all its red neighbours (rows j-2..j)
    updated, so the result is identical to two full passes, but every row comes from main memory
    once per sweep instead of twice."""
    ny = V.shape[0]
    for _ in range(n_sweeps):
        for j in range(1, ny):
            if j < ny - 1:
                _relax_row(V, wx, wy, i_stop, omega, j, 0)
            if j >= 2:
                _relax_row(V, wx, wy, i_stop, omega, j - 1, 1)


def warm_up():
    """Compile (or load from Numba's cache) every compiled function on a tiny grid, so that no
    compile time ends up in a timing.  Call before timing anything."""
    xs, ys = uniform_grid(8.0, 8.0)
    wx, wy, fixed, V = setup(xs, ys)
    sor_sweeps(V, wx, wy, free_ranges(fixed), 1.5, 2)
    energy_S(V, wx, wy)


# ================================================================================================
# SOR driver: checks of S, stopping rules, trajectory log
# ================================================================================================
_TRAJ_FIELDS = ['timestamp', 'method', 'label', 'n_nodes', 'omega', 'sweeps', 'time_s', 'S', 'err_pct']


def _log_trajectory(rows):
    os.makedirs(bc.RESULTS_DIR, exist_ok=True)
    new = not os.path.isfile(TRAJ_CSV)
    with open(TRAJ_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(_TRAJ_FIELDS)
        w.writerows(rows)


def remaining_decrease(S_hist):
    """Geometric extrapolation of how much more S will come down.  Over the last two equal stretches
    of sweeps (each about 5% of the run so far, at least one check interval) S fell by d1 and then
    d2.  If the decrease is geometric, q = d2/d1 per stretch and the rest of the series is
    d2 q / (1 - q).  Returns +inf when the decay is not (yet) geometric (q >= 1), and 0 when S no
    longer changes beyond rounding."""
    n = len(S_hist)
    L = max(1, (n - 1) // 20)
    if n < 2 * L + 1:
        return math.inf
    S0, S1, S2 = S_hist[-1 - 2 * L], S_hist[-1 - L], S_hist[-1]
    d1, d2 = S0 - S1, S1 - S2
    if d2 <= 1e-14 * abs(S2):
        return 0.0
    if d1 <= 0.0 or d2 >= d1:
        return math.inf
    q = d2 / d1
    return d2 * q / (1.0 - q)


def solve_sor(V, wx, wy, fixed, omega, target=None, stop_at_target=True, tol=TOL_REL,
              check_every=CHECK_EVERY, max_sweeps=10 ** 9, time_limit_s=None, method='', label='',
              log=True, progress_s=300.0, quiet=False):
    """Red-black SOR from the potential V (changed in place).  S is evaluated every `check_every`
    sweeps (included in the time).  Stops at the first of:
      (i)   S <= target (if target is given and stop_at_target): records sweeps and time to target;
      (ii)  converged: the extrapolated remaining decrease of S (remaining_decrease) is below tol*S
            at two successive checks;
      (iii) max_sweeps or time_limit_s.
    The trajectory (at most ~250 rows, spaced geometrically in sweeps, plus the last one) is appended
    to results/sor_trajectories.csv.  Call warm_up() first so that no compile time is included.
    Returns a dict: S, sweeps, time_s, reason, hit (sweeps, time_s, S at the first check with
    S <= target, or None), remaining (estimate at the end), omega, check_every, history (S at
    every check, starting with the initial S)."""
    n_nodes = V.size
    i_stop = free_ranges(fixed)
    hist = [energy_S(V, wx, wy)]
    rows, next_log, last_logged = [], 0, -1
    sweeps, small, hit, reason, rem = 0, 0, None, '', math.inf
    t0 = time.perf_counter()
    t_print = t0

    def row(sw, t, S):
        return [datetime.datetime.now().isoformat(timespec='seconds'), method, label, n_nodes,
                f'{omega:.9f}', sw, f'{t:.4f}', f'{S:.10f}', f'{100 * bc.rel_err(S):.6f}']

    rows.append(row(0, 0.0, hist[0]))
    last_logged = 0
    while True:
        sor_sweeps(V, wx, wy, i_stop, omega, check_every)
        sweeps += check_every
        S = energy_S(V, wx, wy)
        t = time.perf_counter() - t0
        hist.append(S)
        if sweeps >= next_log:                             # thin the log: about 4% steps in sweeps
            rows.append(row(sweeps, t, S))
            last_logged = sweeps
            next_log = max(sweeps + check_every, int(sweeps * 1.04))
            if len(rows) >= 20:
                if log:
                    _log_trajectory(rows)
                rows = []
        if target is not None and hit is None and S <= target:
            hit = (sweeps, t, S)
            if stop_at_target:
                reason = 'target reached'
                break
        rem = remaining_decrease(hist)
        small = small + 1 if rem < tol * S else 0
        if small >= 2:
            reason = 'converged'
            break
        if sweeps >= max_sweeps:
            reason = 'max sweeps'
            break
        if time_limit_s is not None and t >= time_limit_s:
            reason = 'time limit'
            break
        if not quiet and time.perf_counter() - t_print >= progress_s:
            t_print = time.perf_counter()
            print(f'      {label}: sweep {sweeps:8d}  S = {S:.8f} ({100 * bc.rel_err(S):+.5f}%)  '
                  f'remaining ~{rem:.1e}  {t / 60:7.1f} min  {1e9 * t / (sweeps * n_nodes):.2f} ns/update',
                  flush=True)
    if log:
        if last_logged != sweeps:
            rows.append(row(sweeps, t, S))
        _log_trajectory(rows)
    return dict(S=S, sweeps=sweeps, time_s=t, reason=reason, hit=hit, remaining=rem, omega=omega,
                check_every=check_every, history=hist)


# ================================================================================================
# One rung of an SOR benchmark (used by bench_sor_graded.py, bench_sor_squares.py, ...rectangles.py)
# ================================================================================================
def run_sor_rung(method, label, make_grid, exact_lambda=False, time_limit_s=None, progress_s=300.0):
    """Build the grid (make_grid() returns xs, ys), choose omega, run SOR from V = 1e-30 (electrode at
    V = 1; see setup) until S <= S_MODEL or convergence, and record the result with bc.record:
      setup_s : grid, bond weights, mask (and the omega formula for uniform grids)
      solve_s : SOR time to the target (S checks included), or to convergence if the grid's own S
                is above the target (beats_model = 0 then)
    exact_lambda=True (graded grids): lambda_min from eigsh, NOT counted in either time (generous to
    SOR); its time is given in the notes.  Returns the dict of solve_sor plus n_nodes, setup_s."""
    t0 = time.perf_counter()
    xs, ys = make_grid()
    wx, wy, fixed, V = setup(xs, ys)
    lam = None if exact_lambda else lambda_uniform_calibrated(xs, ys)
    setup_s = time.perf_counter() - t0
    eig_note = ''
    if exact_lambda:
        t1 = time.perf_counter()
        lam = jacobi_lambda_exact(wx, wy, fixed)
        eig_note = f'; eigsh for rho_J {time.perf_counter() - t1:.1f} s (not counted)'
    omega = omega_sor(lam)
    n = V.size
    print(f'   {label}: {xs.size} x {ys.size} = {n:,} nodes ({25 * n / 1e9:.2f} GB), 1 - rho_J = {lam:.4e}, '
          f'omega = {omega:.9f}; setup {setup_s:.1f} s', flush=True)
    res = solve_sor(V, wx, wy, fixed, omega, target=bc.S_MODEL, time_limit_s=time_limit_s,
                    method=method, label=label, progress_s=progress_s)
    hit = res['hit']
    sweeps, solve_s, S = (hit if hit else (res['sweeps'], res['time_s'], res['S']))
    ns = 1e9 * res['time_s'] / (res['sweeps'] * n)
    notes = (f"{res['reason']}; {sweeps} sweeps; omega {omega:.9f} (2 - 1.15(2 - omega_opt), 1 - rho_J "
             f"{lam:.4e}{' eigsh' if exact_lambda else ' formula x 1.08'}); S checked every "
             f"{res['check_every']} sweeps; {ns:.2f} ns/node-update incl. checks{eig_note}")
    if not hit and res['reason'] == 'converged':
        notes += f'; converged S (remaining < {TOL_REL:g} S) is above the target'
    bc.record(method, label, n, setup_s, solve_s, S, notes=notes)
    res.update(n_nodes=n, setup_s=setup_s, ns_per_update=ns, omega=omega)
    return res
