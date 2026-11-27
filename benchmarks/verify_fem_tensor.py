"""
verify_fem_tensor.py -- checks of fem_tensor.py, and of fem2d.py against it.

fem_tensor.py was written without using fem2d.py's code; this is the only place where the two meet.

    python verify_fem_tensor.py                  (about 30 s here; needs fem2d.py and ide_model.py,
                                                  found by bench_common: set DIELECTRIC_IDES)

Tests (each prints PASS or FAIL):
  A. Element matrices: the energy of x, y, x + 2y, xy (and x^2 for p = 2) on a 0.3 x 1.7 element is
     exact; a wrong local-node ordering would mix up the x and y parts and fail here.
  B. Patch test: a two-layer parallel-plate capacitor (interface on a mesh line, graded mesh, hx != hy).
     The exact V (piecewise linear in y) and psi (linear in x) are in the element space, so the
     primal and dual must both give the exact capacitance, for p = 1 and p = 2.
  C. fem2d.py: bilinear primal on fem2d's meshes (h_c = t_e/16, growth 0.12 and t_e/64, 0.04) vs
     fem2d.FEM2D(...).S(47, 4.53).  The two agree to about 1e-11 relative; what is left is rounding
     in fem2d's final sum V^T K V (long thin elements make it a sum of large terms of both signs).
     To show this, fem2d's own matrices and solution are taken: their V^T K V reproduces fem2d's S,
     and the same V evaluated as a sum of squares at Gauss points (fem_tensor's way) reproduces
     fem_tensor's S to ~1e-13, and the nodal potentials agree to ~1e-12.
  D. Dual (lower bound), bilinear, vs the numbers of an earlier prototype (not included; 7 decimals).
  E. lower <= upper on every mesh (bilinear on fem2d's meshes, biquadratic on p2_mesh meshes).
  F. Thin electrodes (t_e = 0) between two uniform media, biquadratic primal, vs the exact
     conformal-mapping S (ide_model.conformal_mapping_S): above it, and converging to it (the
     tip singularity r^(1/2) is strong, so convergence is slower than for the real geometry).

Test C's tolerance is 3e-11, not 1e-11: on the t_e/64 mesh fem2d's own rounding is 1.5e-11.

Expected output (Claude's sandbox, 21 Nov 2026; about 25 s): every test PASS; C: fem2d/fem_tensor - 1
= +5.7e-12 and +1.5e-11; fem2d's own V summed as squares equals fem_tensor's S to ~1e-16, and the
nodal potentials agree to 4e-12 and 3e-11.
"""
import bench_common as bc       # first: pins numpy/scipy to one thread

import time

import numpy as np
import scipy.sparse.linalg as spla

import fem2d
import fem_tensor as ft
from ide_model import IDEGeometry, conformal_mapping_S

GEOM = dict(s=bc.S_GAP, w=bc.W_FINGER, te=bc.T_E, tp=bc.T_P, y_extent_in_a=bc.Y_EXTENT_IN_A)
EF, ES = bc.EPS_FILM, bc.EPS_SUB
failures = []


def check(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        failures.append(name)


def test_element_matrices():
    print('A. Element matrices: energy of simple functions on a 0.3 x 1.7 element (coefficient 1)')
    hx, hy = 0.3, 1.7
    for p in (1, 2):
        KX, KY = ft.element_matrices(p)
        Ke = (hy / hx) * KX + (hx / hy) * KY
        xi = np.tile(np.linspace(0, 1, p + 1), p + 1)          # local node (a, b): index a + (p+1) b
        eta = np.repeat(np.linspace(0, 1, p + 1), p + 1)
        x, y = hx * xi, hy * eta
        cases = [('x', x, hx * hy), ('y', y, hx * hy), ('x + 2y', x + 2 * y, 5 * hx * hy),
                 ('xy', x * y, hx * hy * (hx ** 2 + hy ** 2) / 3)]
        if p == 2:
            cases.append(('x^2', x ** 2, 4 * hx ** 3 * hy / 3))
        worst = max(abs(f @ Ke @ f / exact - 1) for _, f, exact in cases)
        check(f'p = {p}', worst < 1e-13, f"{', '.join(c[0] for c in cases)}: worst relative error {worst:.1e}")


def test_patch():
    print('B. Patch test: two-layer parallel plate (eps 47 under eps 4.53), exact answer')
    Lx, d1, d, e1, e2 = 3.0, 0.7, 2.0, EF, ES
    W_exact = Lx / (d1 / e1 + (d - d1) / e2)          # capacitance per unit length, eps0 = 1
    xl = ft.graded_lines([0.0, 1.1, Lx], [0.5, 0.05, 0.3], 0.5, 0.3)
    yl = ft.graded_lines([0.0, d1, d], [0.2, 0.01, 0.4], 0.4, 0.5)
    for p in (1, 2):
        m = ft.TensorMesh(xl, yl, p)
        layer1 = m.J < np.searchsorted(yl, d1)          # elements below the interface
        K1, K2 = m.stiffness(layer1), m.stiffness(~layer1)
        i, j = m.node_grid()
        # primal: V = 0 on the bottom plate, 1 on the top plate; natural (D_x = 0) at the sides
        top = (j == m.NY - 1)
        V, _ = ft.solve_with_fixed_nodes(e1 * K1 + e2 * K2, (j == 0) | top, top.astype(float))
        W = m.energy(V, np.where(layer1, e1, e2))
        # dual: psi = 0 on the left side, 1 on the right side (unit flux); free on the plates
        right = (i == m.NX - 1)
        psi, _ = ft.solve_with_fixed_nodes(K1 / e1 + K2 / e2, (i == 0) | right, right.astype(float))
        W_dual = m.energy(psi, np.where(layer1, 1 / e1, 1 / e2))
        err_p, err_d = W / W_exact - 1, 1 / (W_dual * W_exact) - 1
        check(f'p = {p} ({m.n_nodes} nodes)', max(abs(err_p), abs(err_d)) < 1e-12,
              f'primal {err_p:+.1e}, dual {err_d:+.1e} relative to exact')


def fem2d_potential(F, eps_film, eps_sub):
    """fem2d's own matrix and potential: the same three lines as fem2d.FEM2D.S()."""
    eps = {'Q': eps_sub, 'P': eps_film, 'A': 1.0, 'E': 1.0}
    K = sum(eps[r] * Kr for r, Kr in F.K.items())
    V = F.V_fixed.astype(float)
    V[F.free] = spla.spsolve(K[F.free][:, F.free].tocsc(), -K[F.free][:, F.fixed] @ V[F.fixed])
    return K, V


def test_fem2d():
    print('C. Bilinear primal vs fem2d.py on fem2d\'s meshes (eps_film 47, eps_sub 4.53)')
    g = IDEGeometry(s_um=bc.S_GAP, w_um=bc.W_FINGER, te_um=bc.T_E)
    for div, growth in [(16, 0.12), (64, 0.04)]:
        F = fem2d.FEM2D(g, bc.T_P, hc_um=bc.T_E / div, growth=growth)
        S_fem2d = F.S(EF, ES)
        cell = ft.QuarterCell(p=1, hc=bc.T_E / div, growth=growth, **GEOM)
        S_mine = cell.S(EF, ES, 'primal')
        same_mesh = F.n_nodes == cell.n_nodes
        check(f'h_c = t_e/{div}, growth {growth}', same_mesh and abs(S_mine / S_fem2d - 1) < 3e-11,
              f'{cell.n_nodes:,} nodes; fem2d S = {S_fem2d:.12f}, fem_tensor S = {S_mine:.12f}, '
              f'difference {S_fem2d / S_mine - 1:+.1e}')
        # where the difference comes from: fem2d's own potential, energy evaluated two ways
        K, V = fem2d_potential(F, EF, ES)
        S_uKu = 4 / np.pi * (V @ (K @ V))
        coef = EF * cell.region['film'] + ES * cell.region['sub'] + 1.0 * cell.region['air']
        S_squares = 4 / np.pi * cell.mesh.energy(V, coef)
        dV = np.abs(V - cell.u).max()
        check('   ... fem2d\'s own V', abs(S_squares / S_mine - 1) < 1e-12 and dV < 1e-10,
              f'as V^T K V {S_uKu:.12f} (= fem2d), as a sum of squares {S_squares:.12f} '
              f'(fem_tensor {S_squares / S_mine - 1:+.1e}); max |V_fem2d - V_fem_tensor| = {dV:.1e}')


PROTOTYPE = [(16, 0.12, 30375, 7.1864217), (32, 0.06, 75402, 7.1875381),
             (64, 0.04, 162447, 7.1878281), (128, 0.02, 598502, 7.1880254)]


def test_dual_and_order():
    print('D, E. Bilinear dual vs prototype lower bounds; lower <= upper on the same mesh')
    for div, growth, n_proto, S_proto in PROTOTYPE:
        cell = ft.QuarterCell(p=1, hc=bc.T_E / div, growth=growth, **GEOM)
        lo, up = cell.bounds(EF, ES)
        check(f'D: t_e/{div}, growth {growth}', cell.n_nodes == n_proto and abs(lo - S_proto) <= 5.1e-8,
              f'{cell.n_nodes:,} nodes, lower bound {lo:.9f} (prototype {S_proto:.7f})')
        check('E:    lower <= upper', lo <= up, f'{lo:.9f} <= {up:.9f} (width {up / lo - 1:.2e})')
    for gr in (0.3, 0.2, 0.14):
        cell = ft.QuarterCell(**dict(ft.p2_mesh(gr, bc.T_E), **GEOM))
        lo, up = cell.bounds(EF, ES)
        check(f'E: p2_mesh({gr})', lo <= up,
              f'{cell.n_nodes:,} nodes: {lo:.11f} <= {up:.11f} (width {up / lo - 1:.2e})')


def test_conformal():
    print('F. Thin electrodes (t_e = 0), eps 47 above and 4.53 below, biquadratic primal vs conformal mapping')
    exact = conformal_mapping_S(bc.S_GAP, bc.W_FINGER, EF, ES)
    prev = None
    for div, growth in [(64, 0.1), (256, 0.07)]:
        geom = dict(GEOM, te=0.0, tp=bc.Y_EXTENT_IN_A * bc.A)     # "film" fills the upper half
        cell = ft.QuarterCell(p=2, hc=bc.T_E / div, growth=growth, hx_max=2.0, hy_max=8.0, **geom)
        S = cell.S(EF, ES, 'primal')
        rel = S / exact - 1
        ok = 0 < rel < (1e-5 if prev is None else prev)
        check(f'h_c = {bc.T_E / div * 1000:.2f} nm, growth {growth}', ok,
              f'{cell.n_nodes:,} nodes: S = {S:.9f}, exact {exact:.9f}, FEM/exact - 1 = {rel:+.2e}')
        prev = rel


if __name__ == '__main__':
    bc.banner('verify_fem_tensor.py: checks of fem_tensor.py and of fem2d.py')
    t0 = time.perf_counter()
    test_element_matrices()
    test_patch()
    test_fem2d()
    test_dual_and_order()
    test_conformal()
    print(f"\n{'ALL TESTS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}"
          f'  ({time.perf_counter() - t0:.0f} s)')
