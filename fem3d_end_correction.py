"""
fem3d_end_correction.py   (advanced)
=====================================

Three-dimensional finite-element calculation of the capacitance added by the finger tips and bus
bars, which the 2D model leaves out (Appendix C of McClure et al. 2026).  This is how the rule
N_eff = N - 1/2 + N a / L of Sec. 3.2.2 was obtained.  You need it only if your electrodes differ
substantially from ours and you want to check that rule for them.

    python fem3d_end_correction.py                                   (bare substrate, ~10 s; paper: 1.68%)
    python fem3d_end_correction.py --thickness-nm 437.8 --eps-film 47  (a film: blank and film runs, ~20 s)
    python fem3d_end_correction.py --paper                           (all film types of the paper, ~2 min)

OUTPUT
------
delta  = fractional capacitance that the 2D model (overlap length L, N - 1/2 fingers) leaves out.
For a film, the quantity that matters is the correction to the FILM'S CONTRIBUTION (film capacitance
minus blank capacitance), expressed as an extra overlap length per finger end in units of a.  The
N_eff rule assumes 0.5 a; the paper finds 0.46 a to 0.60 a for our films.

GEOMETRY (lengths in micrometers)
---------------------------------
One half-period cell in x (0 to a), with mirror planes at the finger centers x = 0 (the + finger,
half width w/2) and x = a (the - finger).  z runs from the middle of the overlap (z = 0) outward:
the - finger ends at z = L/2; the + finger continues across a tip gap to the + bus bar (width B),
which spans the whole cell; the chip (substrate and film) ends a distance `chip` beyond the bus bar.
The electrodes are infinitely thin, in the plane y = 0.  Quartz (thickness tq) is below, the film
(thickness tp) above, air elsewhere.  V = 0 on the far boundaries y = +-Y and z = Z.

The analyzer measures the MUTUAL capacitance between the two combs (its guard removes the
capacitance of each comb to ground), so two runs are combined:
  'anti': combs at +1 and -1; at z = 0 the potential is the 2D solution;
  'sym' : both combs at +1 (the common-mode part, i.e. the capacitance to ground);
  mutual delta = delta_anti - delta_sym.

METHOD
------
Trilinear elements on a tensor-product mesh, graded toward the electrode edges, tips and bus-bar
edges (smallest element `he`, growth `g`, largest `hfar`).  Because the dielectric constant depends
only on y and z, the x-direction can be diagonalized with the generalized eigenvectors of the 1D
x-matrices: each x-mode then gives an independent 2D (y, z) problem in each half-space, which is
condensed onto the electrode plane y = 0.  The electrode conditions are imposed on the resulting
dense plane system.  This is exact for the discrete problem and much cheaper than a direct 3D solve.
The default mesh (he = 1 um, g = 0.3) agrees with finer meshes to about 0.001 percentage points.
"""

import argparse
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fem2d import graded_axis

# Parameters of the paper's calculation (lengths in um): half-period a, finger width w, overlap L,
# gap between finger tips and the opposite bus bar, bus-bar width B, chip margin beyond the bus bar,
# quartz thickness tq, film thickness tp (0 = bare substrate), dielectric constants, domain size Y, Z,
# and the mesh: smallest element he, growth g, largest hfar.
PAPER_PARAMS = dict(a=80.05, w=29.78, L=7475.0, gap=400.0, B=1000.0, chip=1000.0, tq=500.0, tp=0.0,
                    eQ=4.6, eP=1.0, Y=15000.0, Z=20000.0, he=1.0, g=0.3, hfar=2000.0)


def _mat1d(xs):
    """1D linear-element stiffness and mass matrices on the nodes xs."""
    n = len(xs)
    K, M = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n - 1):
        h = xs[i + 1] - xs[i]
        K[i:i + 2, i:i + 2] += np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        M[i:i + 2, i:i + 2] += h * np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])
    return K, M


def _mat2d(ys, zs, epsfun):
    """Bilinear stiffness and mass matrices on the (y, z) grid (node index j*nz + k); the dielectric
    constant of each element is epsfun(y_center, z_center)."""
    ny, nz = len(ys), len(zs)
    J, Kk = np.meshgrid(np.arange(ny - 1), np.arange(nz - 1), indexing='ij')
    J, Kk = J.ravel(), Kk.ravel()
    hy, hz = ys[J + 1] - ys[J], zs[Kk + 1] - zs[Kk]
    e = epsfun(0.5 * (ys[J] + ys[J + 1]), 0.5 * (zs[Kk] + zs[Kk + 1]))
    nodes = np.stack([J * nz + Kk, (J + 1) * nz + Kk, (J + 1) * nz + Kk + 1, J * nz + Kk + 1], 1)
    sy = np.array([[1, -1], [-1, 1.0]])
    mm = np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])
    loc = [(0, 0), (1, 0), (1, 1), (0, 1)]
    Ky = np.array([[sy[p[0], q[0]] * mm[p[1], q[1]] for q in loc] for p in loc])
    Kz = np.array([[mm[p[0], q[0]] * sy[p[1], q[1]] for q in loc] for p in loc])
    Mm = np.array([[mm[p[0], q[0]] * mm[p[1], q[1]] for q in loc] for p in loc])
    Ke = e[:, None, None] * ((hz / hy)[:, None, None] * Ky + (hy / hz)[:, None, None] * Kz)
    Me = e[:, None, None] * (hy * hz)[:, None, None] * Mm
    r, c = np.repeat(nodes, 4, 1).ravel(), np.tile(nodes, (1, 4)).ravel()
    N = ny * nz
    return (sp.csr_matrix((Ke.ravel(), (r, c)), shape=(N, N)), sp.csr_matrix((Me.ravel(), (r, c)), shape=(N, N)))


def _grids(p):
    """Mesh lines in x, y, z, refined at the finger edges, material boundaries, finger tips and bus bar."""
    a, w, he, g = p['a'], p['w'], p['he'], p['g']
    xs = graded_axis([0.0, w / 2, a - w / 2, a], [2.0, he, he, 2.0], 4.0, g)
    tp, tq, Y = p['tp'], p['tq'], p['Y']
    yk = {-Y: p['hfar'], -tq: max(he, g * tq * 0.5), 0.0: min(he, tp / 3) if tp > 0 else he, Y: p['hfar']}
    if tp > 0:
        yk[tp] = min(he, tp / 3)
    ks = sorted(yk)
    ys = graded_axis(ks, [yk[k] for k in ks], p['hfar'], g)
    zt = p['L'] / 2                     # tip of the - finger
    zb = zt + p['gap']                  # start of the + bus bar
    zbe = zb + p['B']                   # end of the + bus bar
    zc = zbe + p['chip']                # edge of the chip
    zk = {0.0: 200.0, zt: he, zb: he, zbe: he, zc: min(p['hfar'], max(he, g * p['chip'] * 0.5)), p['Z']: p['hfar']}
    ks = sorted(zk)
    zs = graded_axis(ks, [zk[k] for k in ks], p['hfar'], g)
    return xs, ys, zs, (zt, zb, zbe, zc)


def _eps_yz(p, z_chip):
    """Dielectric constant as a function of (y, z): quartz below, film above (within the chip), air elsewhere."""
    def f(yc, zz):
        e = np.ones_like(yc)
        inchip = zz < z_chip
        e[(yc < 0) & (yc > -p['tq']) & inchip] = p['eQ']
        if p['tp'] > 0:
            e[(yc > 0) & (yc < p['tp']) & inchip] = p['eP']
        return e
    return f


def solve_cell(p, mode='anti', verbose=False):
    """One 3D calculation.  mode 'anti': combs at +1/-1; 'sym': both at +1.
    Returns a dict with delta (fraction the 2D model leaves out) and qp, the 2D charge per unit length
    of the + finger (proportional to the 2D capacitance; used to compare films with the blank)."""
    t0 = time.time()
    xs, ys, zs, (zt, zb, zbe, zc) = _grids(p)
    a, w = p['a'], p['w']
    nx, ny, nz = len(xs), len(ys), len(zs)
    j0 = int(np.argmin(abs(ys)))                       # index of the electrode plane y = 0
    Kx, Mx = _mat1d(xs)
    lam, V = sla.eigh(Kx, Mx)                          # x-modes
    lam[0] = 0.0
    A, Bm = Mx @ V, V.T @ Mx                           # nodal reaction = A @ modal; modal coefficient = Bm @ nodal
    plus_x, minus_x = xs <= w / 2 + 1e-9, xs >= a - w / 2 - 1e-9

    # ---- 2D (x, y) reference on the same x and y lines, with the materials of the overlap region
    ef = _eps_yz(p, zc)
    Ky1, My1 = np.zeros((ny, ny)), np.zeros((ny, ny))
    for jj in range(ny - 1):
        h = ys[jj + 1] - ys[jj]
        e = ef(np.array([0.5 * (ys[jj] + ys[jj + 1])]), np.array([0.0]))[0]
        Ky1[jj:jj + 2, jj:jj + 2] += e * np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        My1[jj:jj + 2, jj:jj + 2] += e * h * np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])
    K2 = (sp.kron(sp.csr_matrix(Kx), sp.csr_matrix(My1)) + sp.kron(sp.csr_matrix(Mx), sp.csr_matrix(Ky1))).tocsr()
    I2, J2 = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    I2, J2 = I2.ravel(), J2.ravel()
    fix2 = (J2 == 0) | (J2 == ny - 1)
    v2 = np.zeros(nx * ny)
    pl, mi = (J2 == j0) & plus_x[I2], (J2 == j0) & minus_x[I2]
    fix2 |= pl | mi
    v2[pl], v2[mi] = 1.0, -1.0
    fr2 = ~fix2
    v2[fr2] = spla.spsolve(K2[fr2][:, fr2].tocsc(), -K2[fr2][:, fix2] @ v2[fix2])
    R2 = K2 @ v2
    qp, qm = R2[pl].sum(), R2[mi].sum()
    phi2 = v2.reshape(nx, ny)

    # ---- the two half-spaces y > 0 and y < 0, condensed onto the plane y = 0, one x-mode at a time
    Stot, Ttot = np.zeros((nx, nz, nz)), np.zeros((nx, nz))
    for half in ('up', 'down'):
        yh, jsel = (ys[j0:], np.arange(j0, ny)) if half == 'up' else (ys[:j0 + 1], np.arange(0, j0 + 1))
        Kyz, Myz = _mat2d(yh, zs, ef)
        nyh = len(yh)
        jp = 0 if half == 'up' else nyh - 1
        Jn, Kn = np.meshgrid(np.arange(nyh), np.arange(nz), indexing='ij')
        Jn, Kn = Jn.ravel(), Kn.ravel()
        isP = Jn == jp                                                      # plane nodes
        isO = ((Jn == nyh - 1) if half == 'up' else (Jn == 0)) | (Kn == nz - 1)   # far boundary, V = 0
        isB = ((Kn == 0) & ~isP) if mode == 'anti' else np.zeros_like(isP)  # z = 0: 2D solution ('anti')
        isI = ~(isP | isO | isB)
        P = np.where(isP)[0][np.argsort(Kn[isP])]
        Bn, I = np.where(isB)[0], np.where(isI)[0]
        gB = phi2[:, jsel[Jn[Bn]]] if mode == 'anti' else np.zeros((nx, len(Bn)))
        gBm = Bm @ gB
        for m in range(nx):
            Am = (Kyz + lam[m] * Myz).tocsr()
            AII = Am[I][:, I].tocsc()
            AIP, API, APP = Am[I][:, P], Am[P][:, I], Am[P][:, P]
            AIB, APB = Am[I][:, Bn], Am[P][:, Bn]
            lu = spla.splu(AII)
            X = lu.solve(np.hstack([AIP.toarray(), (AIB @ gBm[m])[:, None]]))
            Stot[m] += APP.toarray() - API @ X[:, :-1]
            Ttot[m] += APB @ gBm[m] - API @ X[:, -1]
        if verbose:
            print(f'   half-space {half}: grid {nyh} x {nz}, {nx} x-modes ({time.time() - t0:.0f} s)', flush=True)

    # ---- dense system for the potential on the plane y = 0; impose the electrode potentials
    C = A[:, None, :] * Bm.T[None, :, :]
    G = (C.reshape(nx * nx, nx) @ Stot.reshape(nx, nz * nz)).reshape(nx, nx, nz, nz).transpose(0, 2, 1, 3).reshape(nx * nz, nx * nz)
    T = (A @ Ttot).reshape(nx * nz)
    Ii, Kk = np.meshgrid(np.arange(nx), np.arange(nz), indexing='ij')
    Ii, Kk = Ii.ravel(), Kk.ravel()
    zz = zs[Kk]
    plus = (plus_x[Ii] & (zz <= zb + 1e-9)) | ((zz >= zb - 1e-9) & (zz <= zbe + 1e-9))   # + finger and its bus bar
    minus = minus_x[Ii] & (zz <= zt + 1e-9) & ~plus                                      # - finger up to its tip
    Phi = np.zeros(nx * nz)
    if mode == 'anti':
        fix = plus | minus | (Kk == 0) | (Kk == nz - 1)
        Phi[plus], Phi[minus] = 1.0, -1.0
        Phi[Kk == 0] = phi2[Ii[Kk == 0], j0]
    else:
        fix = plus | minus | (Kk == nz - 1)
        Phi[plus], Phi[minus] = 1.0, 1.0
    fr = ~fix
    Phi[fr] = sla.solve(G[np.ix_(fr, fr)], -(T[fr] + G[np.ix_(fr, fix)] @ Phi[fix]), assume_a='gen')
    R = G @ Phi + T
    Qp, Qm = R[plus].sum(), R[minus].sum()
    # delta: 3D charge relative to the 2D charge per unit length times the overlap half-length
    delta = (Qp - Qm) / ((qp - qm) * zt) - 1 if mode == 'anti' else (Qp + Qm) / ((qp - qm) * zt)
    return dict(delta=delta, qp=qp, nodes=(nx, ny, nz), seconds=time.time() - t0)


def end_correction(p, verbose=False):
    """Mutual-capacitance end correction (what the analyzer measures): delta_anti - delta_sym."""
    A = solve_cell(p, 'anti', verbose)
    S = solve_cell(p, 'sym', verbose)
    return dict(delta_mutual=A['delta'] - S['delta'], delta_floating=A['delta'], qp=A['qp'],
                nodes=A['nodes'], seconds=A['seconds'] + S['seconds'])


def film_end_correction(p_blank, t_film_um, eps_film, verbose=False):
    """End correction for a film: runs the bare substrate and the film.  Returns the two deltas and the
    correction to the film's contribution (film minus blank capacitance), also expressed as the
    equivalent extra overlap per finger end in units of a (the N_eff rule assumes 0.5)."""
    b = end_correction(dict(p_blank, tp=0.0, eP=1.0), verbose)
    pf = dict(p_blank, tp=t_film_um, eP=eps_film)
    f = end_correction(pf, verbose)
    r = f['qp'] / b['qp']                               # 2D film / blank capacitance ratio
    d_inc = (f['delta_mutual'] * r - b['delta_mutual']) / (r - 1)
    return dict(delta_blank=b['delta_mutual'], delta_film=f['delta_mutual'], ratio=r, delta_contribution=d_inc,
                extra_per_end_in_a=d_inc * p_blank['L'] / (2 * p_blank['a']))


def paper():
    """Appendix C: bare substrate and the film types of the paper (mesh he = 1 um, g = 0.3)."""
    p = dict(PAPER_PARAMS)
    b = end_correction(p)
    print(f'bare substrate: mutual {100 * b["delta_mutual"]:.3f}%  (paper 1.68%),  floating {100 * b["delta_floating"]:.2f}%', flush=True)
    films = [('MAPbI3', 0.4378, 47.0), ('PEA2PbI4', 0.80628, 10.6), ('PEA2PbI4 t5', 0.657801, 11.0),
             ('BA2PbI4', 1.079, 10.7), ('HA2PbI4', 1.619, 9.3), ('PEA2PbBr4', 1.583158, 6.1), ('PEA2PbCl4', 0.36, 6.1)]
    print('films (paper: 1.0-1.6% with films; 0.46a to 0.60a of extra overlap per end for the films\' contribution)')
    for name, tp, e in films:
        f = end_correction(dict(p, tp=tp, eP=e))
        r = f['qp'] / b['qp']
        d_inc = (f['delta_mutual'] * r - b['delta_mutual']) / (r - 1)
        print(f'  {name:12s} t_p {tp:6.3f} um, eps {e:5.1f}: delta {100 * f["delta_mutual"]:.3f}%, film contribution '
              f'{100 * d_inc:.3f}% = {d_inc * p["L"] / (2 * p["a"]):.2f} a of extra overlap per end', flush=True)


def main(argv=None):
    q = argparse.ArgumentParser(description='3D end correction of the IDE capacitance (Appendix C of McClure et al. 2026).')
    q.add_argument('--paper', action='store_true', help='bare substrate and all film types of the paper')
    q.add_argument('--thickness-nm', type=float, help='film thickness (nm); omit for the bare substrate')
    q.add_argument('--eps-film', type=float, default=10.0, help='film dielectric constant')
    q.add_argument('--eps-sub', type=float, default=PAPER_PARAMS['eQ'], help='substrate dielectric constant')
    for k, d in [('a', 'half-period s + w'), ('w', 'finger width'), ('L', 'overlap length'), ('gap', 'tip-to-bus-bar gap'),
                 ('B', 'bus-bar width'), ('chip', 'chip margin beyond the bus bar'), ('tq', 'substrate thickness')]:
        q.add_argument(f'--{k}-um', type=float, default=PAPER_PARAMS[k], help=f'{d} in um (default {PAPER_PARAMS[k]})')
    q.add_argument('--he-um', type=float, default=1.0, help='smallest element (default 1 um; use ~a/80)')
    q.add_argument('--growth', type=float, default=0.3, help='element growth factor (default 0.3)')
    a = q.parse_args(argv)
    if a.paper:
        paper()
        return
    p = dict(PAPER_PARAMS, a=a.a_um, w=a.w_um, L=a.L_um, gap=a.gap_um, B=a.B_um, chip=a.chip_um, tq=a.tq_um,
             eQ=a.eps_sub, he=a.he_um, g=a.growth)
    if a.thickness_nm is None:
        b = end_correction(p, verbose=True)
        print(f'bare substrate: the ends add {100 * b["delta_mutual"]:.3f}% to the mutual capacitance '
              f'({b["delta_mutual"] * p["L"] / (2 * p["a"]):.2f} a of extra overlap per end)')
    else:
        r = film_end_correction(p, a.thickness_nm / 1000, a.eps_film)
        print(f'blank: {100 * r["delta_blank"]:.3f}%   with the film: {100 * r["delta_film"]:.3f}%')
        print(f'film contribution: {100 * r["delta_contribution"]:.3f}% = {r["extra_per_end_in_a"]:.2f} a of extra overlap '
              f'per end (the N_eff rule assumes 0.5 a)')


if __name__ == '__main__':
    main()
