"""
ide_model.py
============

Capacitance of an interdigitated electrode (IDE) covered by a thin film, and extraction of the
film's dielectric function from a measured capacitance.

This is the model of Section III B of

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


HOW THE MODEL WORKS (Sec. III B 1-5)
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
   effective number of half-periods N_eff = N - 1/2 + N a / L (Sec. III B 2):
     - the "- 1/2" accounts for the two outermost fingers, which have a neighbor on one side only;
     - the "N a / L" accounts for the extra capacitance of the finger tips and bus bars
       (3D finite-element calculations, Appendix C of the paper).
5. To get a film's dielectric constant, we first solve for eps_Q from the capacitance of a
   blank substrate, then solve for eps_P from the capacitance of the substrate with the film.

The model also works with COMPLEX dielectric constants eps = eps' - i eps'' (all expressions are
analytic in eps), which lets us fit the measured complex capacitance Cp - i G/omega (Sec. III B 7).


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
    geom  = IDEGeometry()                 # our IDE: s = 50.27 um, w = 29.78 um, t_e = 0.12 um, ...
    model = IDEModel(geom)
    eps_Q = model.solve_substrate(19.88e-12)                       # blank capacitance (F) -> eps_Q
    eps_P = model.solve_film(24.36e-12, eps_Q, t_film_um=0.4378)   # film capacitance (F) -> eps_P
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

EPS0 = 8.8541878128e-12  # vacuum permittivity, F/m


# --------------------------------------------------------------------------------------------
# 1. Geometry
# --------------------------------------------------------------------------------------------
@dataclass
class IDEGeometry:
    """Dimensions of the interdigitated electrodes.

    The default values are those of the IDEs in the paper (Sec. II).  Change them for your own
    electrodes, e.g.  IDEGeometry(s_um=10.0, w_um=10.0, te_um=0.1, N=50, L_mm=2.0).

    s_um  : gap between adjacent electrodes, in micrometers
    w_um  : width of each electrode (finger), in micrometers
    te_um : electrode thickness, in micrometers (0.12 um = 10 nm Cr + 100 nm Au, as measured)
    N     : number of fingers (both combs together)
    L_mm  : length over which adjacent fingers overlap, in millimeters
    N_eff : effective number of half-periods.  Leave as None to use the rule of Sec. III B 2,
            N_eff = N - 1/2 + N a / L.  Set a number to override it (for example N - 1/2 to
            leave out the finger-tip and bus-bar correction, or a value from your own 3D
            calculation).
    """
    s_um: float = 50.27
    w_um: float = 29.78
    te_um: float = 0.120
    N: int = 130
    L_mm: float = 7.475
    N_eff: Optional[float] = None

    @property
    def a_um(self) -> float:
        """Half-period a = s + w (um): the distance between the centers of neighboring fingers."""
        return self.s_um + self.w_um

    def n_eff(self) -> float:
        """Effective number of half-periods (Sec. III B 2)."""
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
    per node until it reaches h_max_um.  For our electrodes (s = 50.27 um, h_edge = 0.015 um)
    this gives 42 nodes.
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
# 3. The model
# --------------------------------------------------------------------------------------------
class IDEModel:
    """Capacitance of the IDE with a film, using the energy-minimizing in-plane potential.

    Create one IDEModel per electrode geometry and reuse it: setting it up takes a fraction of a
    second, and each capacitance evaluation then takes about 10 ms.

    Parameters
    ----------
    geom      : IDEGeometry
    n_terms   : number of Fourier terms in the sums over n (the paper uses 20,000).
    h_edge_um : smallest node spacing of f(x), at the electrode edge (default t_e/8).
    h_max_um  : largest node spacing (default min(2 um, s/25)).
    growth    : growth factor of the node spacing away from the edge.

    The defaults reproduce the paper.  The capacitance changes by less than 1e-5 (relative) if
    the node spacings are halved or the number of terms is doubled.
    """

    def __init__(self, geom: IDEGeometry = None, n_terms: int = 20000, h_edge_um: float = None,
                 h_max_um: float = None, growth: float = 0.15):
        self.geom = geom if geom is not None else IDEGeometry()
        g = self.geom
        self.s, self.w, self.a, self.te = g.s_um, g.w_um, g.a_um, g.te_um
        if h_edge_um is None:
            h_edge_um = self.te / 8.0 if self.te > 0 else self.s / 30000.0
        if h_max_um is None:
            h_max_um = min(2.0, self.s / 25.0)

        # Fourier index: m = 2n - 1 = 1, 3, 5, ...  and wavenumber k_n = (2n - 1) pi / a
        self.m = 2.0 * np.arange(1, n_terms + 1) - 1.0
        self.k = self.m * np.pi / self.a

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

        # Eq. (14): B_n = 4/(a k_n^2) * sum_i sigma_i [sin(k_n x_i) - sin(k_n x_{i-1})].
        # Since the slopes are linear in the nodal values, B = A f with the matrix A below.
        kk = self.k[:, None]
        per_slope = 4.0 / (self.a * kk ** 2) * (np.sin(kk * self.x[None, 1:]) - np.sin(kk * self.x[None, :-1]))
        self.A = per_slope @ D                                # shape (n_terms, n_nodes)

        # Region 3 (Eq. 12, evaluated exactly from the slopes): the integral of f'(x)^2 over one
        # whole gap is 2 * sum_i sigma_i^2 dx_i (f is odd about the gap center).  As a quadratic
        # form in the nodal values f this is f^T R f with:
        self.R = D.T @ np.diag(2.0 * self.dx) @ D

    # ---- the layered dielectric above the electrodes ---------------------------------------
    def eps_above(self, eps_film, d_um):
        """Effective dielectric constant 'seen' by Fourier mode n looking up from the top of the
        electrodes: a film of thickness d (= t_p - t_e) with air above it (Eqs. 5, 6, 9, 10).

            eps_above,n = eps_P (1 - r exp(-2 k_n d)) / (1 + r exp(-2 k_n d)),  r = (eps_P-1)/(eps_P+1)

        It tends to eps_P for a thick film (d >> a) and to 1 (air) for d -> 0.
        Works for real or complex eps_film.
        """
        if d_um <= 0:
            return np.ones_like(self.k)
        r = (eps_film - 1.0) / (eps_film + 1.0)
        E = np.exp(-2.0 * self.k * d_um)
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
        # Regions 1 and 2: weight of each Fourier mode = eps_sub (below) + eps_above (above)
        weight = eps_sub + self.eps_above(eps_film, t_film_um - self.te)
        # Quadratic form S(f) = f^T H f, with H = A^T diag(m * weight) A + (2/pi) eps_film t_e R
        H = (self.A * (self.m * weight)[:, None]).T @ self.A + (2.0 / np.pi) * eps_film * self.te * self.R
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
        """Substrate dielectric constant eps_Q from the capacitance (F) of a blank IDE (Sec. III B 7)."""
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
        capacitance is C* = Cp - i G / omega (Sec. III B 7)."""
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
# 4. Reference results used to check the model (Sec. III B 6)
# --------------------------------------------------------------------------------------------
def conformal_mapping_S(s_um, w_um, eps_above, eps_below):
    """Exact dimensionless capacitance for infinitely thin electrodes between two uniform media
    (conformal mapping): C = eps0 (eps_above + eps_below) N L K(k') / (2 K(k)), k = sin(pi s / 2a),
    k' = sqrt(1 - k^2), with K the complete elliptic integral of the first kind."""
    from scipy.special import ellipk
    a = s_um + w_um
    k2 = np.sin(np.pi * s_um / (2 * a)) ** 2            # scipy's ellipk takes the parameter m = k^2
    return (eps_above + eps_below) * 4.0 * ellipk(1 - k2) / (np.pi * ellipk(k2))


def den_otter_S(model: IDEModel, eps_film, eps_sub, t_film_um):
    """Dimensionless capacitance with den Otter's FIXED in-plane potential, Eq. (3):
    B_n = 4 / ((2n-1) pi) * J0((2n-1) pi s / (2a)).  Regions 1 and 2 only (its Region 3 sum
    diverges, Sec. III B 5), so use it with te_um = 0.  For comparison only."""
    from scipy.special import j0
    B = 4.0 / (model.m * np.pi) * j0(model.m * np.pi * model.s / (2 * model.a))
    weight = eps_sub + model.eps_above(eps_film, t_film_um - model.te)
    return np.sum(model.m * weight * B ** 2)
