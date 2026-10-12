"""
fem2d.py
========

Two-dimensional finite-element (FEM) calculation of the IDE capacitance, used in the paper to check
the model of ide_model.py (Sec. III B 6 and Appendix C of McClure et al. 2026).

The FEM solves the electrostatics problem  div( eps grad V ) = 0  directly on a mesh, with no
assumption about the in-plane potential, so comparing it with the model tests the model's
approximations (mainly the Region-3 potential between the electrodes).

    python fem2d.py --eps-film 47 --eps-sub 4.52 --thickness-nm 438     (one comparison, ~10 s)
    python fem2d.py --blank --eps-sub 4.52                              (bare substrate)
    python fem2d.py --paper            (the comparisons of Sec. III B 6 and Supplemental S4, S9; ~3 min)

Use it to check the model for YOUR geometry, especially if your electrodes are thick compared with
the half-period a (the model's error grows roughly in proportion to t_e/a; Sec. III B 6).


THE CALCULATION (Appendix C)
----------------------------
* One quarter of a period, by symmetry: x runs from the center of a gap (x = 0, where V = 0 by
  antisymmetry) to the center of an electrode (x = a/2, where the normal derivative of V is zero).
* The electrode (s/2 < x < a/2, 0 < y < t_e) is held at V = 1.  Quartz fills y < 0; the film fills
  the gap and covers the electrode up to y = t_p; air is above.
* V = 0 at y = +-3a.  Every Fourier component of the potential decays at least as exp(-pi|y|/a), so
  the energy beyond that boundary is about 1e-8 of the total.
* Bilinear rectangular elements on a mesh whose lines follow the electrode surfaces and material
  boundaries.  The mesh is graded toward the electrode corners, where the field is singular: the
  smallest elements are h_c = t_e/16 (default) across, and element sizes grow by `growth` (0.12)
  times the distance from the corners, up to 0.5 um in x and 4 um in y.
* The capacitance is computed from the stored energy (Eq. 1) and reported as the same dimensionless
  S as ide_model.py, so  C = geom.prefactor() * S.
* The FEM solution is itself an admissible trial potential, so its S is an upper bound that
  DECREASES as the mesh is refined.  With h_c = t_e/64 and growth 0.04 (about 1.6e5 nodes) the
  model lies about 0.01% above the FEM for our electrodes.
"""

import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ide_model import IDEGeometry, IDEModel

# Element stiffness matrices of a bilinear rectangle (nodes counterclockwise from lower left),
# for the x- and y-derivative parts; scaled by hy/hx and hx/hy respectively.
KX = np.array([[2, -2, -1, 1], [-2, 2, 1, -1], [-1, 1, 2, -2], [1, -1, -2, 2]]) / 6.0
KY = np.array([[2, 1, -1, -2], [1, 2, -2, -1], [-1, -2, 2, 1], [-2, -1, 1, 2]]) / 6.0


def graded_axis(keys, h_at_keys, h_max, growth):
    """Mesh lines along one axis.  `keys` are coordinates that must be mesh lines (material
    boundaries, electrode edges); near key k the element size is h_at_keys[k] and it grows by
    `growth` times the distance from the nearest key, up to h_max."""
    pts = [keys[0]]
    for A, B, hA, hB in zip(keys[:-1], keys[1:], h_at_keys[:-1], h_at_keys[1:]):
        x, seg = A, []
        while True:
            h = min(h_max, hA + growth * (x - A), hB + growth * (B - x))
            if x + 1.3 * h >= B:
                break
            x += h
            seg.append(x)
        pts += seg + [B]
    return np.array(pts)


class FEM2D:
    """Quarter-period FEM of an IDE with a film of thickness t_film_um (from the substrate surface).

    The mesh is built once; S() can then be evaluated for any dielectric constants.
    Set blank=True for a bare substrate (air between and above the electrodes)."""

    def __init__(self, geom: IDEGeometry = None, t_film_um: float = None, blank: bool = False,
                 hc_um: float = None, growth: float = 0.12, y_extent_in_a: float = 3.0,
                 hx_max_um: float = 0.5, hy_max_um: float = 4.0):
        g = geom if geom is not None else IDEGeometry()
        s, a, te = g.s_um, g.a_um, g.te_um
        if te <= 0:
            raise ValueError('fem2d.py needs electrodes of finite thickness (te_um > 0)')
        tp = te if (blank or t_film_um is None) else t_film_um
        if tp < te - 1e-12:
            raise ValueError('the film must fill the gaps: t_film_um >= te_um')
        hc = hc_um if hc_um is not None else te / 16.0
        Y = y_extent_in_a * a
        # mesh lines: x at the gap center, electrode edge, electrode center; y at the far boundaries,
        # the substrate surface, the electrode top and the film top
        xk = {0.0: hx_max_um, s / 2: hc, a / 2: hx_max_um}
        xs = graded_axis(sorted(xk), [xk[k] for k in sorted(xk)], hx_max_um, growth)
        yk = {-Y: hy_max_um, 0.0: hc, te: hc, Y: hy_max_um}
        if tp > te + 1e-9:
            yk[tp] = 4 * hc
        ys = graded_axis(sorted(yk), [yk[k] for k in sorted(yk)], hy_max_um, growth)
        nx, ny = len(xs), len(ys)
        self.n_nodes = nx * ny

        # elements and their regions: Q = substrate, P = film, A = air, E = electrode
        I, J = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing='ij')
        I, J = I.ravel(), J.ravel()
        hx, hy = xs[I + 1] - xs[I], ys[J + 1] - ys[J]
        xc, yc = 0.5 * (xs[I] + xs[I + 1]), 0.5 * (ys[J] + ys[J + 1])
        in_gap = xc < s / 2
        region = np.full(len(xc), 'A')
        region[yc < 0] = 'Q'
        film = (in_gap & (yc > 0) & (yc < tp)) | (~in_gap & (yc > te) & (yc < tp))
        if blank:
            film[:] = False
        region[film] = 'P'
        region[(~in_gap) & (yc > 0) & (yc < te)] = 'E'
        nodes = np.stack([J * nx + I, J * nx + I + 1, (J + 1) * nx + I + 1, (J + 1) * nx + I], 1)
        Ke = (hy / hx)[:, None, None] * KX + (hx / hy)[:, None, None] * KY
        # one stiffness matrix per region, so that K(eps) = sum_r eps_r K_r is cheap to rebuild
        self.K = {}
        for r in 'QPAE':
            m = region == r
            if m.any():
                nd, ke = nodes[m], Ke[m]
                rows, cols = np.repeat(nd, 4, 1).ravel(), np.tile(nd, (1, 4)).ravel()
                self.K[r] = sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        # boundary conditions: V = 0 at the gap center and the far boundaries, V = 1 on the electrode
        X, Yg = np.meshgrid(xs, ys)
        X, Yg = X.ravel(), Yg.ravel()
        fixed = (X < 1e-12) | (Yg < -Y + 1e-9) | (Yg > Y - 1e-9)
        electrode = (X >= s / 2 - 1e-12) & (Yg >= -1e-12) & (Yg <= te + 1e-12)
        fixed |= electrode
        self.V_fixed = np.where(electrode, 1.0, 0.0)
        self.free, self.fixed = ~fixed, fixed

    def S(self, eps_film=1.0, eps_sub=4.53):
        """Dimensionless capacitance (C = geom.prefactor() * S) for the given dielectric constants.
        Complex values are allowed (eps' - 1j*eps'')."""
        eps = {'Q': eps_sub, 'P': eps_film, 'A': 1.0, 'E': 1.0}
        K = sum(eps[r] * Kr for r, Kr in self.K.items())
        V = self.V_fixed.astype(np.result_type(eps_film, eps_sub, float))
        Kff = K[self.free][:, self.free].tocsc()
        V[self.free] = spla.spsolve(Kff, -K[self.free][:, self.fixed] @ V[self.fixed])
        # Stored energy per unit length of the quarter cell: (eps0/2) V^T K V.  One half-period (one gap,
        # electrodes at +1 and -1) holds two such cells, so its capacitance per unit length is
        # c = 2U / (Delta V)^2 = eps0 V^T K V / 2.  With C = eps0 N L (pi/8) S this gives:
        return (4.0 / np.pi) * (V @ (K @ V))


def compare(geom, eps_film, eps_sub, t_um, blank=False, hc_um=None, growth=0.12):
    """Model and FEM for one case; returns (S_model, S_fem, model/FEM - 1, number of FEM nodes)."""
    model = IDEModel(geom)
    fem = FEM2D(geom, t_um, blank=blank, hc_um=hc_um, growth=growth)
    sm = model.S_blank(eps_sub) if blank else model.S(eps_film, eps_sub, t_um)
    sf = fem.S(1.0 if blank else eps_film, eps_sub)
    return sm, sf, sm / sf - 1, fem.n_nodes


def paper():
    """Reproduce the model-versus-FEM comparisons of Sec. III B 6 and Supplemental sections S4 and S9."""
    g = IDEGeometry()
    te = g.te_um
    print('S9: mesh convergence, MAPbI3-like film (eps 47, 0.4378 um) on quartz (eps 4.53)')
    print('    (paper: model/FEM - 1 = -0.033, -0.012, +0.004, +0.008% for 18444, 30240, 74061, 160165 nodes)')
    for hc, gr in [(te / 8, 0.20), (te / 16, 0.12), (te / 32, 0.06), (te / 64, 0.04)]:
        sm, sf, d, n = compare(g, 47.0, 4.53, 0.4378, hc_um=hc, growth=gr)
        print(f'    h_c = t_e/{te / hc:.0f}, growth {gr}: {n:7d} nodes, model/FEM - 1 = {100 * d:+.4f}%', flush=True)
    fine = dict(hc_um=te / 64, growth=0.04)              # the most refined mesh of the paper
    model = IDEModel(g)
    fem = FEM2D(g, 0.4378, **fine)
    print('\nS4(a): 0.44 um film, most refined mesh  (paper: +0.0122, +0.0121, +0.0091, +0.0079, +0.0075, +0.0087%)')
    for e in [2, 6, 20, 47, 200, 1000]:
        print(f'    eps_film {e:6.0f}: model/FEM - 1 = {100 * (model.S(e, 4.53, 0.4378) / fem.S(e, 4.53) - 1):+.4f}%', flush=True)
    # Sec. III B 6: use FEM capacitances as "measured data" (a blank and films like ours) and recover
    # the dielectric constants with the model, exactly as for real data: eps_Q from the blank first.
    print('\nRecovery test (Sec. III B 6, paper: eps_P recovered to within 0.07%):')
    blank = FEM2D(g, blank=True, **fine)
    q = model.solve_substrate(g.prefactor() * blank.S(1.0, 4.53))
    print(f'    eps_Q 4.53 recovered from the FEM blank as {q:.4f}')
    for e, tp in [(6.1, 0.36), (6.1, 1.583), (10.5, 0.806), (10.6, 1.079), (9.3, 1.619), (47.0, 0.4378)]:
        er = model.solve_film(g.prefactor() * FEM2D(g, tp, **fine).S(e, 4.53), q, tp)
        print(f'    eps_P {e:5.1f}, t_p = {tp:5.3f} um: recovered {er:8.4f} ({100 * (er / e - 1):+.3f}%)', flush=True)
    print('\nS4(b): bare substrate, thicker electrodes (model error grows with t_e/a; Sec. III B 6)')
    print('    (SI figure: +0.010, +0.107, +0.52%; for thick electrodes those meshes were not fully converged)')
    for te_a in [0.0015, 0.0125, 0.06]:
        gt = IDEGeometry(te_um=te_a * g.a_um)
        for div, gr in [(64, 0.04), (128, 0.03)]:
            sm, sf, d, n = compare(gt, 1.0, 4.53, None, blank=True, hc_um=gt.te_um / div, growth=gr)
            print(f'    t_e/a = {te_a:.4f}, h_c = t_e/{div}, growth {gr}: {n:6d} nodes, model/FEM - 1 = {100 * d:+.3f}%', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description='2D finite-element check of the IDE model (Appendix C of McClure et al. 2026).')
    p.add_argument('--paper', action='store_true', help='reproduce Supplemental sections S4 and S9')
    p.add_argument('--eps-film', type=float, default=47.0, help='film dielectric constant (default 47)')
    p.add_argument('--eps-sub', type=float, default=4.53, help='substrate dielectric constant (default 4.53)')
    p.add_argument('--thickness-nm', type=float, default=437.8, help='film thickness from the substrate, nm')
    p.add_argument('--blank', action='store_true', help='bare substrate (no film)')
    p.add_argument('--s-um', type=float, default=50.27); p.add_argument('--w-um', type=float, default=29.78)
    p.add_argument('--te-um', type=float, default=0.12)
    p.add_argument('--refine', type=int, default=16, help='smallest element = t_e / this (default 16; paper finest 64)')
    p.add_argument('--growth', type=float, default=0.12, help='element growth factor (default 0.12; paper finest 0.04)')
    a = p.parse_args(argv)
    if a.paper:
        paper()
        return
    g = IDEGeometry(s_um=a.s_um, w_um=a.w_um, te_um=a.te_um)
    t_um = None if a.blank else a.thickness_nm / 1000
    sm, sf, d, n = compare(g, a.eps_film, a.eps_sub, t_um, blank=a.blank, hc_um=a.te_um / a.refine, growth=a.growth)
    print(f'FEM ({n} nodes): S = {sf:.7f}   model: S = {sm:.7f}   model/FEM - 1 = {100 * d:+.4f}%')
    print('(The FEM value decreases toward the true capacitance as the mesh is refined: try --refine 64 --growth 0.04.)')


if __name__ == '__main__':
    main()
