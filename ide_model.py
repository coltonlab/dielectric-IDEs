"""
ide_model.py
============

Capacitance of an interdigitated electrode (IDE) covered by a thin film, and extraction of the
film's dielectric function from a measured capacitance.

This is the model of Section 3.2 of

    C. E. McClure, K. R. Hansen, C. M. Shirley, C. Lindsay, L. Whittaker-Brooks, J. S. Colton,
    "Measuring Dielectric Functions of Thin Films Using Interdigitated Electrodes, with
    Application to Metal Halide Perovskites" (2026).

Equation numbers in the comments refer to that paper.

Most users will not call this module directly; they will run extract_permittivity.py, which
reads measured data files and calls the functions below.  Read this file if you want to know
what the code does, or if you want to use the model from your own Python scripts.


GEOMETRY (Fig. 3 of the paper; cross section perpendicular to the electrode fingers)
-----------------------------------------------------------------------------------

              air (dielectric constant 1)
    y = t_p   ---------------------------------------------------------------  top of the film
              film, eps_P                                  (Region 1a)
    y = t_e   ......[ electrode ]..........[ electrode ].........  top of the electrodes
              film  [  +V0      ]   film   [   -V0     ]   film    (Region 3)
    y = 0     ======[===========]==========[===========]======== substrate surface
              substrate, eps_Q  (much thicker than a/pi)          (Region 2)

    w   = electrode (finger) width
    s   = gap between adjacent electrodes
    a   = s + w = half-period (center-to-center distance of neighboring fingers)
    t_e = electrode thickness
    t_p = film thickness, measured from the substrate surface (so the film is t_p - t_e thick
          above the electrodes).  The model assumes t_p >= t_e: the film fills the gaps.
    N   = number of fingers;  L = length over which adjacent fingers overlap.

For a bare ("blank") substrate, use eps_P = 1 and t_p = t_e (Geometry 3 of the paper).
For a thick liquid covering the electrodes (e.g. water), use a very large t_p (e.g. 1 cm).


HOW THE MODEL WORKS (Sec. 3.2.1-3.2.5)
-------------------------------------
1. The potential in the plane of the electrodes, f(x), is +V0 on one set of fingers and -V0 on
   the other.  In each gap it is piecewise linear on about 40 nodes (closest together at the
   electrode edges), and its Fourier coefficients B_n follow from the slopes (Eq. 14).
2. Above and below that plane the potential is the exact solution of Laplace's equation for
   the layered dielectrics (substrate below; film then air above), Eqs. (4)-(6).  Between the
   electrodes (Region 3) the potential is f(x), extended unchanged in y (Eq. 7).
3. The stored energy, expressed as a capacitance, is then a quadratic function of the nodal
   values of f(x) (Eqs. 8-12).  We choose the nodal values that MINIMIZE the energy, which by
   Dirichlet's principle gives the best upper bound on the capacitance within this family of
   trial potentials.  The optimal f(x) depends on the dielectric constants, so it is recomputed
   for every trial value.
4. The capacitance of the whole device is the capacitance per half-period, times L, times an
   effective number of half-periods N_eff = N - 1/2 + N a / L (Sec. 3.2.2):
     - the "- 1/2" accounts for the two outermost fingers, which have a neighbor on one side only;
     - the "N a / L" accounts for the extra capacitance of the finger tips and bus bars
       (3D finite-element calculations, Appendix C of the paper).
5. To get a film's dielectric constant, we first solve for eps_Q from the capacitance of a
   blank substrate, then solve for eps_P from the capacitance of the substrate with the film.

The model also works with COMPLEX dielectric constants eps = eps' - i eps'' (all expressions are
analytic in eps), which lets us fit the measured complex capacitance Cp - i G/omega (Sec. 3.2.7).


THE SUMS OVER FOURIER MODES
---------------------------
Regions 1 and 2 contribute a sum over the Fourier modes n = 1, 2, 3, ... (Eqs. 9-11).  The code
evaluates it to infinity, exactly, and fast:
  * Mode n feels the top of the film only through exp(-2 k_n d), with d = t_p - t_e.  Once that
    factor is below 1e-17, the mode sees an infinitely thick film: its weight is eps_Q + eps_P to
    double precision.  For the films of the paper this holds for every mode after the first 170
    to 1020 (for water, for all modes).
  * So the sum is (eps_Q + eps_P) times the sum over ALL modes with weight 1, plus corrections for
    the first modes.  The sum over all modes depends only on the electrodes and the nodes of f(x),
    so it is computed once, when an IDEModel is created, from a closed form (section 3 below).
  * Each capacitance then needs only the first modes and a linear solve of about 40 equations.
Versions of this code before November 2026 cut the sums off after 20,000 terms.  That made S lower
by 2e-8 to 4e-8 (relative) for the films of the paper and by 7.5e-8 for the blank substrates,
which changes none of the values printed in the paper; for infinitely thin electrodes (t_e = 0)
the finest node spacings were not resolved, and S could fall below the exact value.
IDEModel(geom, n_terms=20000) reproduces the cut-off sums.


UNITS
-----
Lengths: micrometers (um) inside the model; the overlap length L is given in mm.
Capacitance: farads.  Frequencies: Hz.

Internally the code works with a dimensionless capacitance S, defined by

    C = (eps0 * N_eff * L * pi / 8) * S                                      (cf. Eqs. 9-13)

so that S = sum_n (2n-1) [eps_Q + eps_above,n] B_n^2  +  (2/pi) eps_P t_e * integral of f'(x)^2
over one gap.  The first sum is Regions 1 and 2 (Eqs. 9-11), the second term is Region 3 (Eq. 12).


QUICK EXAMPLE
-------------
    from ide_model import IDEGeometry, IDEModel
    geom  = IDEGeometry()                 # our IDE: s = 50.27 um, w = 29.78 um, t_e = 0.115 um, ...
    model = IDEModel(geom)
    eps_Q = model.solve_substrate(19.88e-12)                       # blank capacitance (F) -> eps_Q
    eps_P = model.solve_film(24.36e-12, eps_Q, t_film_um=0.4378)   # film capacitance (F) -> eps_P
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.special import zeta

EPS0 = 8.8541878128e-12  # vacuum permittivity, F/m


# --------------------------------------------------------------------------------------------
# 1. Geometry
# --------------------------------------------------------------------------------------------
@dataclass
class IDEGeometry:
    """Dimensions of the interdigitated electrodes.

    The default values are those of the IDEs in the paper (Sec. 2).  Change them for your own
    electrodes, e.g.  IDEGeometry(s_um=10.0, w_um=10.0, te_um=0.1, N=50, L_mm=2.0).

    s_um  : gap between adjacent electrodes, in micrometers
    w_um  : width of each electrode (finger), in micrometers
    te_um : electrode thickness, in micrometers (0.115 um = 15 nm Cr + 100 nm Au, as deposited)
    N     : number of fingers (both combs together)
    L_mm  : length over which adjacent fingers overlap, in millimeters
    N_eff : effective number of half-periods.  Leave as None to use the rule of Sec. 3.2.2,
            N_eff = N - 1/2 + N a / L.  Set a number to override it (for example N - 1/2 to
            leave out the finger-tip and bus-bar correction, or a value from your own 3D
            calculation).
    """
    s_um: float = 50.27
    w_um: float = 29.78
    te_um: float = 0.115
    N: int = 130
    L_mm: float = 7.475
    N_eff: Optional[float] = None

    @property
    def a_um(self) -> float:
        """Half-period a = s + w (um): the distance between the centers of neighboring fingers."""
        return self.s_um + self.w_um

    def n_eff(self) -> float:
        """Effective number of half-periods (Sec. 3.2.2)."""
        if self.N_eff is not None:
            return float(self.N_eff)
        # N - 1/2 : the two outermost fingers have a neighbor on one side only (2D finite arrays)
        # N a / L : finger tips and bus bars, equivalent to lengthening each overlap by about
        #           a/2 at each end (3D finite-element calculations, Appendix C)
        return self.N - 0.5 + self.N * self.a_um / (self.L_mm * 1000.0)

    def prefactor(self) -> float:
        """Farads per unit of the dimensionless capacitance S:  C = prefactor * S."""
        return EPS0 * self.n_eff() * (self.L_mm * 1e-3) * np.pi / 8.0


# Dimensions of the IDEs used in the paper.  The main wafer batch is the default above; the
# substrate of the HA2PbI4 film (and the second blank) came from a second wafer batch.
PAPER_MAIN_BATCH = IDEGeometry(s_um=50.27, w_um=29.78)
PAPER_SECOND_BATCH = IDEGeometry(s_um=50.13, w_um=29.99)


# --------------------------------------------------------------------------------------------
# 2. Nodes for the piecewise-linear in-plane potential f(x)
# --------------------------------------------------------------------------------------------
def half_gap_nodes(s_um: float, h_edge_um: float, h_max_um: float = 2.0, growth: float = 0.15):
    """Nodes on the half-gap 0 <= x <= s/2 (x = 0 at the center of the gap, x = s/2 at the
    electrode edge), closest together at the electrode edge, where the field is strongest.

    Starting at the edge, the spacing begins at h_edge_um and grows by a factor (1 + growth)
    per node until it reaches h_max_um.  For our electrodes (s = 50.27 um, h_edge = t_e/8 = 0.0144 um)
    this gives 43 nodes.
    """
    half = s_um / 2.0
    dist = [0.0]                       # distances measured from the electrode edge
    h = h_edge_um
    while dist[-1] + h < half:
        dist.append(dist[-1] + h)
        h = min(h * (1.0 + growth), h_max_um)
    x = half - np.array(dist)[::-1]    # convert to x, increasing, ending at the edge x = s/2
    if x[0] > 1e-9:                    # make sure the gap center x = 0 is a node
        x = np.concatenate([[0.0], x])
    x[0] = 0.0
    return np.unique(x)


# --------------------------------------------------------------------------------------------
# 3. The sum over ALL Fourier modes, in closed form
# --------------------------------------------------------------------------------------------
# Regions 1 and 2 contribute  sum_n m_n w_n B_n^2  to S  (m_n = 2n - 1, weight w_n = eps_Q + eps_above,n).
# Beyond the first few hundred modes every weight is the same, eps_Q + eps_P (see IDEModel.S), so we
# need the sum with weight 1 over ALL modes, as a quadratic form in the nodal values of f(x).
# With Eq. (14) and k_n = m pi / a:
#
#     sum_n m_n B_n^2 = (16 a^2 / pi^4) sum_{i,j} sigma_i sigma_j G_ij,
#     G_ij = sum over odd m of [sin(m u_i) - sin(m u_{i-1})] [sin(m u_j) - sin(m u_{j-1})] / m^3,
#
# where sigma_i is the slope of f(x) on segment i (from node i-1 to node i) and u_i = pi x_i / a.
# A product of two sines is a difference of two cosines, and
#
#     sum over odd m of cos(m t) / m^3 = C3(t) - C3(2t)/8,   C3(t) = sum_{n>=1} cos(n t) / n^3,
#
# (the even m are the n = 2, 4, ... terms of C3(t), i.e. C3(2t)/8).  C3 has the expansion
#
#     C3(t) = zeta(3) + (t^2/2) ln t - (3/4) t^2 - sum_{k>=1} zeta(2k) t^(2k+2) / (k (2k+1) (2k+2) (2 pi)^(2k))
#
# for 0 <= t <= pi.  (It follows by integrating once the standard expansion of the Clausen function,
# sum_n sin(n t)/n^2 = t - t ln t + sum_{k>=1} zeta(2k) t^(2k+1) / (k (2k+1) (2 pi)^(2k)), since
# dC3/dt = -sum_n sin(n t)/n^2.)  The terms of the last sum fall by at least a factor 4 each, so
# 30 of them give full double precision.

_K = np.arange(1, 31)
_C3_COEF = zeta(2.0 * _K) / (_K * (2 * _K + 1) * (2 * _K + 2) * (2 * np.pi) ** (2 * _K))


def _c3(t):
    """C3(t) - zeta(3), where C3(t) = sum_{n>=1} cos(n t) / n^3, for real t (numpy arrays welcome).
    Accurate to about 1e-16.  The constant zeta(3) is left out because it cancels in the matrix
    of all_modes_matrix() below, and leaving it out avoids rounding error there."""
    t = np.abs(np.asarray(t, dtype=float)) % (2.0 * np.pi)
    t = np.where(t > np.pi, 2.0 * np.pi - t, t)        # C3 is even and 2 pi-periodic: map t into [0, pi]
    u = t * t
    series = np.full_like(t, _C3_COEF[-1])             # sum_k coef_k u^(k-1), by Horner's rule
    for c in _C3_COEF[-2::-1]:
        series = series * u + c
    log_term = np.zeros_like(t)
    pos = t > 0
    log_term[pos] = 0.5 * u[pos] * np.log(t[pos])     # (t^2/2) ln t, which is 0 at t = 0
    return log_term - 0.75 * u - u * u * series


def all_modes_matrix(x, a):
    """Matrix T of the quadratic form  f^T T f = sum over ALL n >= 1 of (2n-1) B_n^2,  where B_n are
    the Fourier coefficients (Eq. 14) of the piecewise-linear f(x) with nodal values f on the nodes
    x (x[0] = 0 at the gap center, x[-1] = s/2 at the electrode edge) and a is the half-period."""
    x = np.asarray(x, dtype=float)
    u = np.pi * x / a
    odd = lambda t: _c3(t) - _c3(2.0 * t) / 8.0                     # sum over odd m of cos(m t)/m^3 (minus a constant)
    up, uq = u[:, None], u[None, :]
    # Nn[p, q] = (16 a^2 / pi^4) sum over odd m of sin(m u_p) sin(m u_q) / m^3
    Nn = (8.0 * a * a / np.pi ** 4) * (odd(up - uq) - odd(up + uq))
    G = Nn[1:, 1:] - Nn[1:, :-1] - Nn[:-1, 1:] + Nn[:-1, :-1]         # (16 a^2/pi^4) G_ij, segments i, j
    dx = np.diff(x)
    n = len(x)
    D = np.zeros((n - 1, n))                                          # nodal values -> slopes
    D[np.arange(n - 1), np.arange(n - 1)] = -1.0 / dx
    D[np.arange(n - 1), np.arange(1, n)] = 1.0 / dx
    T = D.T @ G @ D
    return 0.5 * (T + T.T)                                            # symmetric up to rounding; make it exact


# --------------------------------------------------------------------------------------------
# 4. The model
# --------------------------------------------------------------------------------------------
class IDEModel:
    """Capacitance of the IDE with a film, using the energy-minimizing in-plane potential.

    Create one IDEModel per electrode geometry and reuse it: setting it up takes a few
    milliseconds, and each capacitance evaluation then takes about 0.2 ms (one core of a 2.1 GHz
    Xeon, for the paper's electrodes and films; 20 to 30 times less than before November 2026).

    Parameters
    ----------
    geom      : IDEGeometry
    n_terms   : None (default): the sums over Fourier modes n run to infinity, exactly (module
                docstring, "THE SUMS OVER FOURIER MODES").  An integer: cut the sums off after
                that many terms instead, as versions of this code before November 2026 did
                (n_terms=20000).  Use it only to reproduce older numbers.
    h_edge_um : smallest node spacing of f(x), at the electrode edge (default t_e/8).
    h_max_um  : largest node spacing (default min(2 um, s/25)).
    growth    : growth factor of the node spacing away from the edge.

    The capacitance changes by about 1e-5 (relative) if the node spacings are halved: for a
    MAPbI3-like film (eps 47, 438 nm) on quartz, by -1.2e-5 when h_edge, h_max and growth are all
    halved, and by -1.1e-5 when only growth is halved.

    Attributes used by other scripts: x, dx (nodes of f and their spacings), s, w, a, te, geom,
    and m, k, A: the indices m = 2n - 1, the wavenumbers k_n and the rows B_n = A[n-1] @ f of the
    modes stored explicitly.  With n_terms=None these are the first modes only (1024 to start
    with, more if a thinner film needs them); the rest are in the closed-form matrix T.
    """

    N_HEAD_INIT = 1024        # modes stored explicitly at setup (for our electrodes: enough if t_p - t_e > 0.25 um)
    N_HEAD_MAX = 100000       # never store more (for our electrodes this limit is reached only if t_p - t_e < 2.5 nm,
                              # and the modes beyond it then change S by less than 1e-9)
    E_CUT = 1e-17             # a mode with exp(-2 k_n d) below this sees an infinitely thick film

    def __init__(self, geom: IDEGeometry = None, n_terms: Optional[int] = None, h_edge_um: float = None,
                 h_max_um: float = None, growth: float = 0.15):
        self.geom = geom if geom is not None else IDEGeometry()
        g = self.geom
        self.s, self.w, self.a, self.te = g.s_um, g.w_um, g.a_um, g.te_um
        self.n_terms = None if n_terms is None else int(n_terms)
        if h_edge_um is None:
            h_edge_um = self.te / 8.0 if self.te > 0 else self.s / 30000.0
        if h_max_um is None:
            h_max_um = min(2.0, self.s / 25.0)

        # Nodes x_0 = 0 (gap center, f = 0) ... x_M = s/2 (electrode edge, f = 1, i.e. V0 = 1)
        self.x = half_gap_nodes(self.s, h_edge_um, h_max_um, growth)
        self.dx = np.diff(self.x)
        n_nodes = len(self.x)

        # D maps the nodal values f_0..f_M to the slopes sigma_i = (f_i - f_{i-1}) / dx_i
        n_seg = n_nodes - 1
        D = np.zeros((n_seg, n_nodes))
        D[np.arange(n_seg), np.arange(n_seg)] = -1.0 / self.dx
        D[np.arange(n_seg), np.arange(1, n_nodes)] = 1.0 / self.dx
        self.D = D

        # Region 3 (Eq. 12, evaluated exactly from the slopes): the integral of f'(x)^2 over one
        # whole gap is 2 * sum_i sigma_i^2 dx_i (f is odd about the gap center).  As a quadratic
        # form in the nodal values f this is f^T R f with:
        self.R = D.T @ np.diag(2.0 * self.dx) @ D

        # Regions 1 and 2.  T is the quadratic form of sum_n m_n B_n^2 with weight 1 for every mode:
        # over ALL modes in closed form (n_terms=None), or over the first n_terms modes.  It depends
        # only on the geometry and the nodes, so it is computed once, here.
        self.m = np.zeros(0)
        self.k = np.zeros(0)
        self.A = np.zeros((0, n_nodes))
        if self.n_terms is None:
            self._store_modes(self.N_HEAD_INIT)
            self.T = all_modes_matrix(self.x, self.a)
        else:
            self._store_modes(self.n_terms)
            self.T = (self.A * self.m[:, None]).T @ self.A

    def _store_modes(self, n):
        """Store the first n Fourier modes: m = 2n - 1, k_n = m pi / a, and the rows of the matrix A
        with B = A f.  Modes already stored are kept; only the new ones are computed."""
        n_old = len(self.m)
        if n <= n_old:
            return
        m = 2.0 * np.arange(n_old + 1, n + 1) - 1.0
        k = m * np.pi / self.a
        # Eq. (14): B_n = 4/(a k_n^2) * sum_i sigma_i [sin(k_n x_i) - sin(k_n x_{i-1})].
        # Since the slopes are linear in the nodal values, B = A f with the matrix A below.
        kk = k[:, None]
        per_slope = 4.0 / (self.a * kk ** 2) * (np.sin(kk * self.x[None, 1:]) - np.sin(kk * self.x[None, :-1]))
        self.m = np.concatenate([self.m, m])
        self.k = np.concatenate([self.k, k])
        self.A = np.vstack([self.A, per_slope @ self.D])                  # shape (n, n_nodes)

    def _n_film_modes(self, d_um):
        """Number of leading modes whose weight depends on the film thickness d = t_p - t_e:
        those with exp(-2 k_n d) >= E_CUT, i.e. n <= (ln(1/E_CUT) a / (2 pi d) + 1) / 2.
        Stores more modes if needed."""
        d_um = max(d_um, 1e-200)                       # (no overflow for absurdly thin films)
        n1 = int(min(np.floor((np.log(1.0 / self.E_CUT) * self.a / (2.0 * np.pi * d_um) + 1.0) / 2.0), 1e12))
        if self.n_terms is not None:
            return max(0, min(n1, self.n_terms))
        n1 = max(0, min(n1, self.N_HEAD_MAX))
        if n1 > len(self.m):
            self._store_modes(min(max(n1, 2 * len(self.m)), self.N_HEAD_MAX))
        return n1

    # ---- the layered dielectric above the electrodes ---------------------------------------
    def eps_above(self, eps_film, d_um, k=None):
        """Effective dielectric constant 'seen' by Fourier mode n looking up from the top of the
        electrodes: a film of thickness d (= t_p - t_e) with air above it (Eqs. 5, 6, 9, 10).

            eps_above,n = eps_P (1 - r exp(-2 k_n d)) / (1 + r exp(-2 k_n d)),  r = (eps_P-1)/(eps_P+1)

        It tends to eps_P for a thick film (d >> a) and to 1 (air) for d -> 0.
        Works for real or complex eps_film.  k: the wavenumbers (default: the stored modes, self.k).
        """
        k = self.k if k is None else np.asarray(k)
        if d_um <= 0:
            return np.ones_like(k)
        r = (eps_film - 1.0) / (eps_film + 1.0)
        E = np.exp(-2.0 * k * d_um)
        return eps_film * (1.0 - r * E) / (1.0 + r * E)

    # ---- the dimensionless capacitance S ------------------------------------------------------
    def S(self, eps_film, eps_sub, t_film_um, return_profile=False):
        """Dimensionless capacitance S (C = geom.prefactor() * S) for a film of dielectric
        constant eps_film and thickness t_film_um (measured from the substrate) on a substrate
        of dielectric constant eps_sub.

        eps_film and eps_sub may be complex (eps' - 1j*eps'').  For real values the result is the
        minimum over all piecewise-linear f(x), i.e. the model's upper bound on the capacitance;
        for complex values it is the analytic continuation of that expression.

        If return_profile is True, also returns the optimized nodal values of f(x) on self.x.
        """
        if t_film_um < self.te - 1e-12:
            raise ValueError('The model assumes the film fills the gaps: t_film_um must be >= te_um.')
        d = t_film_um - self.te
        # Regions 1 and 2: S(f) = f^T [sum_n m_n w_n a_n a_n^T] f, with a_n the rows of A and the weight
        # w_n = eps_sub (below) + eps_above,n (above).
        if d <= 0:
            # no film above the electrodes: every mode sees air above, w_n = eps_sub + 1
            H = (eps_sub + 1.0) * self.T
        else:
            # w_n = (eps_sub + eps_film) + delta_n, with delta_n = eps_above,n - eps_film
            #     = -2 eps_film r E_n / (1 + r E_n),  E_n = exp(-2 k_n d).
            # For n beyond n1, E_n < 1e-17 and delta_n is below the rounding error of w_n (these
            # modes see an infinitely thick film), so they are all in (eps_sub + eps_film) T.
            # Only the first n1 modes need the correction delta_n.
            n1 = self._n_film_modes(d)
            A1, m1, k1 = self.A[:n1], self.m[:n1], self.k[:n1]
            r = (eps_film - 1.0) / (eps_film + 1.0)
            E = np.exp(-2.0 * k1 * d)
            delta = -2.0 * eps_film * r * E / (1.0 + r * E)
            H = (eps_sub + eps_film) * self.T + (A1 * (m1 * delta)[:, None]).T @ A1
        # Region 3: (2/pi) eps_film t_e f^T R f
        H = H + (2.0 / np.pi) * eps_film * self.te * self.R
        # Minimize with the boundary values f_0 = 0 (gap center) and f_M = 1 (electrode edge):
        # setting the derivative with respect to the free nodal values to zero gives a small
        # linear system (about 40 equations).
        n = H.shape[0]
        free = np.arange(1, n - 1)
        f = np.zeros(n, dtype=H.dtype)
        f[-1] = 1.0
        f[free] = np.linalg.solve(H[np.ix_(free, free)], -H[free, n - 1])
        S = f @ H @ f
        return (S, f) if return_profile else S

    def S_blank(self, eps_sub):
        """Dimensionless capacitance of a blank substrate (air above and between the electrodes)."""
        return self.S(1.0, eps_sub, self.te)

    def capacitance(self, eps_film, eps_sub, t_film_um):
        """Capacitance in farads of the whole IDE with the film (Eq. 8, times N_eff and L)."""
        return self.geom.prefactor() * self.S(eps_film, eps_sub, t_film_um)

    def blank_capacitance(self, eps_sub):
        """Capacitance in farads of the whole IDE on a bare substrate (Eq. 13)."""
        return self.geom.prefactor() * self.S_blank(eps_sub)

    # ---- inversion: measured capacitance -> dielectric constant ------------------------------
    def solve_substrate(self, C_blank, lo=1.0, hi=1000.0):
        """Substrate dielectric constant eps_Q from the capacitance (F) of a blank IDE (Sec. 3.2.7)."""
        target = C_blank / self.geom.prefactor()
        return brentq(lambda q: self.S_blank(q) - target, lo, hi, xtol=1e-11)

    def solve_film(self, C_film, eps_sub, t_film_um, lo=1.0001, hi=1e6):
        """Film dielectric constant eps_P from the capacitance (F) of the IDE with the film,
        given the substrate's eps_sub (from solve_substrate) and the film thickness in um.
        Raises ValueError if no eps_P in [lo, hi] matches (e.g. if C_film is below the blank)."""
        target = C_film / self.geom.prefactor()
        return brentq(lambda e: self.S(e, eps_sub, t_film_um) - target, lo, hi, xtol=1e-10)

    def solve_substrate_complex(self, Cp, G, freq_hz, eps_start=None):
        """Complex substrate dielectric constant eps' - i eps'' from the parallel capacitance Cp (F)
        and conductance G (S) of a blank IDE at frequency freq_hz.  The measured complex
        capacitance is C* = Cp - i G / omega (Sec. 3.2.7)."""
        target = (Cp - 1j * G / (2 * np.pi * freq_hz)) / self.geom.prefactor()
        x0 = eps_start if eps_start is not None else self.solve_substrate(Cp)
        return _newton_complex(lambda q: self.S_blank(q + 0j), target, complex(x0))

    def solve_film_complex(self, Cp, G, freq_hz, eps_sub, t_film_um, eps_start=None):
        """Complex film dielectric constant eps' - i eps'' from the film's Cp (F) and G (S).
        eps_sub is the (complex) substrate value from solve_substrate_complex.  The starting
        guess defaults to the Cp-only value with a small loss."""
        target = (Cp - 1j * G / (2 * np.pi * freq_hz)) / self.geom.prefactor()
        if eps_start is None:
            eps_start = self.solve_film(Cp, np.real(eps_sub), t_film_um) - 0.1j
        return _newton_complex(lambda e: self.S(e, eps_sub + 0j, t_film_um), target, complex(eps_start))


def _newton_complex(fun, target, x0, tol=1e-10, max_iter=40):
    """Solve fun(x) = target for complex x by Newton's method.  fun is analytic, so its complex
    derivative can be taken by a central difference along the real axis."""
    x = complex(x0)
    for _ in range(max_iter):
        F = fun(x) - target
        h = 1e-6 * max(1.0, abs(x))
        dF = (fun(x + h) - fun(x - h)) / (2 * h)
        step = F / dF
        x -= step
        if abs(step) < tol * max(1.0, abs(x)):
            return x
    raise RuntimeError('Newton iteration did not converge')


# --------------------------------------------------------------------------------------------
# 5. Reference results used to check the model (Sec. 3.2.6)
# --------------------------------------------------------------------------------------------
def conformal_mapping_S(s_um, w_um, eps_above, eps_below):
    """Exact dimensionless capacitance for infinitely thin electrodes between two uniform media
    (conformal mapping): C = eps0 (eps_above + eps_below) N L K(k') / (2 K(k)), k = sin(pi s / 2a),
    k' = sqrt(1 - k^2), with K the complete elliptic integral of the first kind."""
    from scipy.special import ellipk
    a = s_um + w_um
    k2 = np.sin(np.pi * s_um / (2 * a)) ** 2            # scipy's ellipk takes the parameter m = k^2
    return (eps_above + eps_below) * 4.0 * ellipk(1 - k2) / (np.pi * ellipk(k2))


def den_otter_S(model: IDEModel, eps_film, eps_sub, t_film_um, n_terms=None):
    """Dimensionless capacitance with den Otter's FIXED in-plane potential, Eq. (3):
    B_n = 4 / ((2n-1) pi) * J0((2n-1) pi s / (2a)).  Regions 1 and 2 only (its Region 3 sum
    diverges, Sec. 3.2.5), so use it with te_um = 0.  For comparison only.
    The sum runs over the first n_terms modes: by default the model's n_terms, or 20,000 (as in
    the paper) for a model with exact sums.  The terms fall off only as 1/n^2 here."""
    from scipy.special import j0
    if n_terms is None:
        n_terms = model.n_terms if model.n_terms is not None else 20000
    m = 2.0 * np.arange(1, n_terms + 1) - 1.0
    B = 4.0 / (m * np.pi) * j0(m * np.pi * model.s / (2 * model.a))
    weight = eps_sub + model.eps_above(eps_film, t_film_um - model.te, k=m * np.pi / model.a)
    return np.sum(m * weight * B ** 2)
