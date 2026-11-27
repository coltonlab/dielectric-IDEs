"""
fem_tensor.py -- finite elements of order p = 1 (bilinear) or p = 2 (biquadratic) on a graded
tensor-product mesh of the IDE quarter cell.  Gives a rigorous UPPER bound on the dimensionless
capacitance S (primal formulation, Dirichlet's principle) or a rigorous LOWER bound (dual
formulation, Thomson's principle).

Written independently of fem2d.py so that it can check it (only verify_fem_tensor.py imports fem2d).
Needs numpy and scipy only.  Lengths in micrometres; eps0 = 1.

THE PROBLEM (Supplemental Section S4 of the paper)
    One quarter of a period: 0 < x < a/2, -Y < y < Y, with a = s + w and Y = 3a.
    x = 0    : centre of a gap; V = 0 there by antisymmetry.
    x = a/2  : centre of an electrode; dV/dx = 0 there by mirror symmetry.
    y = +-Y  : V = 0 (the potential decays like exp(-pi |y| / a).  Cutting it off at 3a raises S by
               1.0e-8 relative: with the boundary at 4a or 5a, S is lower by 1.04e-8 (biquadratic
               bounds, Nov 2026).  All bounds below are bounds for this truncated problem.)
    Electrode: s/2 < x < a/2, 0 < y < t_e, held at V = 1.
    Materials: substrate (eps_sub) for y < 0; film (eps_film) for 0 < y < t_p outside the electrode;
               air (eps 1) above y = t_p.
    W = integral of eps |grad V|^2 over the quarter cell is the capacitance per unit length of the
    quarter cell (charge for unit voltage), and S = (4/pi) W is the normalization of fem2d.py and
    ide_model.py (C = geom.prefactor() * S).

TWO VARIATIONAL PRINCIPLES, ONE SET OF MATRICES
    Primal (Dirichlet): for ANY potential V that obeys the boundary conditions, W[V] >= W_true.
        The FEM finds the V of lowest W within the finite-element space, so
        S_upper = (4/pi) W[V_h] >= S_true.
    Dual (Thomson): write the displacement field with a stream function psi,
        D = (d psi/dy, -d psi/dx).
        Then div D = 0 automatically, and the flux of D across any curve is the difference of psi
        between its ends.  On the symmetry plane x = a/2 no flux may cross (D_x = d psi/dy = 0), so
        psi is constant along each of its two pieces outside the electrode: psi = 0 below
        (y <= 0) and psi = 1 above (y >= t_e).  Then exactly one unit of flux leaves the electrode.
        On the boundaries where V is prescribed (x = 0, y = +-Y, the electrode surface) psi is free.
        For ANY such psi, W*[psi] = integral |D|^2 / eps = integral |grad psi|^2 / eps >= 1 / W_true
        (Thomson's principle for unit charge), so S_lower = (4/pi) / W*[psi_h] <= S_true.
    Both energies are integrals of c |grad u|^2 with a coefficient c that is constant in each
    material region: c = eps (primal) or c = 1/eps (dual).  So we build, once per mesh, the matrix
    of integral |grad u|^2 over each region (substrate, film, air), and combine them for any
    dielectric constants:  K = sum_r c_r K_r.  The electrode interior carries no field and is
    left out of both energies (in the primal V = 1 on all its nodes anyway).
    Both bounds stay valid whatever the rounding errors of the linear solve: any admissible trial
    function gives a bound, and a slightly wrong V or psi only loosens it (one step of iterative
    refinement changes S by ~1e-15).  The energy of the computed V or psi is evaluated as a sum
    of squares at Gauss points (TensorMesh.energy), accurate to ~1e-15.

ELEMENTS
    Lagrange elements of order p on rectangles, built from 1D shape functions L_a(xi) on [0, 1]
    with nodes xi = 0, 1/p, ..., 1.  The 2D shape function of local node (a, b) is L_a(xi) L_b(eta),
    local index a + (p+1) b (x index fastest).  For an element hx by hy:
        integral dN_i/dx dN_j/dx = (hy/hx) M1[b,b'] S1[a,a'] = (hy/hx) kron(M1, S1)[i, j]
        integral dN_i/dy dN_j/dy = (hx/hy) S1[b,b'] M1[a,a'] = (hx/hy) kron(S1, M1)[i, j]
    with M1 = integral L_a L_a' and S1 = integral L_a' L_a'' on [0, 1] (numpy's kron(B, A) has entry
    B[b,b'] A[a,a'] at row a + (p+1) b, so the SECOND factor acts on the x index).  M1 and S1 are
    computed by (p+1)-point Gauss-Legendre quadrature, exact for polynomials of degree 2p+1.

MESH
    Mesh lines follow fem2d.py's recipe (re-implemented in graded_lines): key lines x = 0, s/2, a/2
    and y = -Y, 0, t_e, t_p, Y; element size h_c at x = s/2, y = 0 and y = t_e (the electrode
    corners, where the field is singular), 4 h_c at y = t_p, hx_max / hy_max at the outer keys;
    the size grows by `growth` times the distance from the nearest key, capped at hx_max / hy_max.
    For p = 2 these lines are the element boundaries and the mid-side and centre nodes are added.
    Options beyond fem2d's recipe, for p = 2: separate sizes at y = 0 (h_0) and y = t_p (h_tp), and
    a y cap that grows away from the electrode plane (y_far).  p2_mesh(g) is the family of p = 2
    meshes used for the reference bracket (reference_bounds.py) and the d2 benchmark.

LINEAR SOLVER: see solve_spd().

Usage
    import fem_tensor as ft
    cell = ft.QuarterCell(p=1, hc=0.115 / 16, growth=0.12)          # fem2d.py's default mesh
    cell = ft.QuarterCell(**ft.p2_mesh(0.1))                          # biquadratic, 0.59M nodes
    print(cell.n_nodes, cell.S(47, 4.53, 'dual'), cell.S(47, 4.53, 'primal'))   # lower, upper
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# The paper's IDE and film (um); the benchmark scripts pass bench_common's values explicitly.
S_GAP, W_FINGER, T_E, T_P = 50.27, 29.78, 0.115, 0.4378


# ---- 1D shape functions and the element matrices ----------------------------------------------
def shape_functions_1d(p):
    """The 1D Lagrange shape functions of order p on [0, 1] (nodes 0, 1/p, ..., 1) at the p + 1
    Gauss-Legendre points: values L[a, g], derivatives dL[a, g], and the Gauss weights wg[g]."""
    nodes = np.linspace(0.0, 1.0, p + 1)
    xg, wg = np.polynomial.legendre.leggauss(p + 1)      # Gauss points and weights on [-1, 1]
    xg, wg = 0.5 * (xg + 1.0), 0.5 * wg                  # ... mapped to [0, 1]
    L = np.empty((p + 1, p + 1))
    dL = np.empty((p + 1, p + 1))
    for a in range(p + 1):
        others = np.delete(nodes, a)                     # L_a vanishes at every other node ...
        La = np.polynomial.Polynomial.fromroots(others) / np.prod(nodes[a] - others)   # ... and is 1 at its own
        L[a], dL[a] = La(xg), La.deriv()(xg)
    return L, dL, wg


def matrices_1d(p):
    """Mass matrix M1[a, a'] = integral L_a L_a' and stiffness matrix S1[a, a'] = integral L_a' L_a''
    on [0, 1], by Gauss quadrature (exact: the integrands have degree <= 2p)."""
    L, dL, wg = shape_functions_1d(p)
    M1, S1 = (L * wg) @ L.T, (dL * wg) @ dL.T
    # symmetric in exact arithmetic; make them symmetric to the last bit, so that the assembled
    # matrices are exactly symmetric too (solve_spd relies on it)
    return 0.5 * (M1 + M1.T), 0.5 * (S1 + S1.T)


def element_matrices(p):
    """KX, KY: the x- and y-derivative parts of the stiffness matrix of a unit square element,
    local node index a + (p+1) b.  The element matrix of an hx-by-hy element with coefficient c is
    c * ((hy/hx) KX + (hx/hy) KY)."""
    M1, S1 = matrices_1d(p)
    return np.kron(M1, S1), np.kron(S1, M1)


# ---- mesh lines -----------------------------------------------------------------------------------
def graded_lines(keys, h_keys, h_max, growth, far=None):
    """Coordinates of the mesh lines along one axis.

    keys   : increasing coordinates that must be mesh lines (material boundaries, electrode faces)
    h_keys : element size wanted at each key
    Between two keys A < B the size at x is min(h_max, hA + growth (x - A), hB + growth (B - x)),
    i.e. it grows linearly with the distance from the nearer key (geometric grading, the right
    thing near a corner singularity, where the field varies on the scale of the distance from the
    corner).  Lines are laid down one element at a time from A; once the next element would end
    within 1.3 elements of B, the remaining interval becomes the last element.
    far    : None for fem2d.py's recipe.  Otherwise a length L: the cap itself then grows with the
             distance from 0, h_max (1 + |x| / L).  Used for y: far from the electrode plane the
             potential just decays smoothly (like exp(-pi |y| / a) and faster) and its energy
             density keeps falling, so ever coarser elements are good enough there.
    """
    lines = [keys[0]]
    for A, B, hA, hB in zip(keys[:-1], keys[1:], h_keys[:-1], h_keys[1:]):
        x = A
        while True:
            cap = h_max if far is None else h_max * (1.0 + abs(x) / far)
            h = min(cap, hA + growth * (x - A), hB + growth * (B - x))
            if x + 1.3 * h >= B:
                break
            x += h
            lines.append(x)
        lines.append(B)
    return np.array(lines)


# ---- a tensor-product mesh and its stiffness matrices ----------------------------------------------
class TensorMesh:
    """Rectangular elements between consecutive mesh lines xl (x) and yl (y); Lagrange order p.

    Nodes form an NX by NY grid, NX = p (len(xl) - 1) + 1; node (i, j) has number j * NX + i.
    Element (I, J) lies between xl[I], xl[I+1] and yl[J], yl[J+1]; its local node (a, b) is the
    global node (p I + a, p J + b)."""

    def __init__(self, xl, yl, p):
        self.p, self.xl, self.yl = p, np.asarray(xl, float), np.asarray(yl, float)
        self.x = self._add_inner_nodes(self.xl)          # node coordinates along x and y
        self.y = self._add_inner_nodes(self.yl)
        self.NX, self.NY = len(self.x), len(self.y)
        self.n_nodes = self.NX * self.NY
        # element indices (I fastest) and sizes
        I, J = np.meshgrid(np.arange(len(xl) - 1), np.arange(len(yl) - 1))
        self.I, self.J = I.ravel(), J.ravel()
        self.hx = np.diff(self.xl)[self.I]
        self.hy = np.diff(self.yl)[self.J]
        # global node numbers of each element's local nodes, local index a + (p+1) b
        a = np.tile(np.arange(p + 1), p + 1)
        b = np.repeat(np.arange(p + 1), p + 1)
        self.conn = (p * self.J[:, None] + b) * self.NX + (p * self.I[:, None] + a)
        self.KX, self.KY = element_matrices(p)
        # For the energy: d/dxi and d/deta of the shape functions L_a(xi) L_b(eta) at the Gauss
        # points (row: Gauss point g + (p+1) h, at xi_g, eta_h; column: local node a + (p+1) b),
        # and the Gauss weights wg[g] wg[h].  Same Kronecker ordering as the element matrices.
        L, dL, wg = shape_functions_1d(p)
        self.Gx, self.Gy, self.wq = np.kron(L.T, dL.T), np.kron(dL.T, L.T), np.kron(wg, wg)

    def _add_inner_nodes(self, lines):
        """Element boundaries plus p - 1 equally spaced nodes inside each element."""
        p = self.p
        nodes = np.empty(p * (len(lines) - 1) + 1)
        nodes[::p] = lines
        for k in range(1, p):
            nodes[k::p] = lines[:-1] + (k / p) * np.diff(lines)
        return nodes

    def node_grid(self):
        """Integer grid indices (i, j) of every node, in node-number order."""
        i, j = np.meshgrid(np.arange(self.NX), np.arange(self.NY))
        return i.ravel(), j.ravel()

    def stiffness(self, element_mask, chunk=100_000):
        """Matrix of the quadratic form  integral |grad u|^2  over the selected elements
        (coefficient 1).  Assembled in chunks of elements to limit the memory of the triplets."""
        elems = np.flatnonzero(element_mask)
        K = sp.csr_matrix((self.n_nodes, self.n_nodes))
        for e in np.array_split(elems, max(1, len(elems) // chunk)):
            ke = ((self.hy / self.hx)[e, None, None] * self.KX
                  + (self.hx / self.hy)[e, None, None] * self.KY)        # (n_e, nloc, nloc)
            rows = np.broadcast_to(self.conn[e][:, :, None], ke.shape)
            cols = np.broadcast_to(self.conn[e][:, None, :], ke.shape)
            K = K + sp.csr_matrix((ke.ravel(), (rows.ravel(), cols.ravel())), shape=K.shape)
        return K

    def energy(self, u, coef, chunk=200_000):
        """integral of coef |grad u|^2 for nodal values u and one coefficient per element.

        Computed as a sum of non-negative terms, coef * |grad u|^2 * weight at the Gauss points of
        each element (exact for these polynomials), rather than as u^T K u: for long thin elements
        u^T K u is a sum of large terms of both signs that cancel, which costs about 1e-11 in S."""
        total = 0.0
        elems = np.flatnonzero(coef != 0)
        for e in np.array_split(elems, max(1, len(elems) // chunk)):
            ue = u[self.conn[e]]                         # (elements, local nodes)
            ue = ue - ue[:, :1]                          # a constant shift leaves grad u unchanged
            ux = (ue @ self.Gx.T) / self.hx[e, None]     # du/dx at the Gauss points
            uy = (ue @ self.Gy.T) / self.hy[e, None]
            total += np.sum(coef[e] * self.hx[e] * self.hy[e] * ((ux ** 2 + uy ** 2) @ self.wq))
        return total


# ---- linear solve --------------------------------------------------------------------------------
ORDERING = 'MMD_AT_PLUS_A'


def reduce_to_free(K, fixed, u_fixed):
    """Setting the derivative of u^T K u with respect to each free nodal value to zero, with u =
    u_fixed on the fixed nodes, gives  A u_free = rhs  with A = K[free, free] (CSR) and
    rhs = -K[free, fixed] u_fixed."""
    free, fix = np.flatnonzero(~fixed), np.flatnonzero(fixed)
    Kf = K[free]                                         # rows of the free nodes
    return Kf[:, free], -(Kf[:, fix] @ u_fixed[fix])


def solve_spd(A, rhs, ordering=None):
    """Solve A x = rhs for a sparse symmetric positive definite A (CSR); returns (x, info).

    Solver: SuperLU (scipy's splu).  A needs no pivoting for stability, so we tell SuperLU to keep
    the diagonal pivots (diag_pivot_thresh = 0, SymmetricMode) and to order the unknowns by
    minimum degree on the symmetric structure of A (permc_spec = 'MMD_AT_PLUS_A'): in effect a
    Cholesky factorization with a fill-reducing symmetric ordering.  On these meshes this beats
    scipy's default (COLAMD, a column ordering meant for unsymmetric matrices with partial
    pivoting): 1.7 to 2.3 times fewer nonzeros in the factors, 1.6 to 2.8 times faster
    factorization, 1.5 to 2.5 times less peak memory (p = 1 and p = 2 meshes of 0.16 to 0.6
    million nodes; see the notes in reference_bounds.py).
    A is exactly symmetric, so its CSR arrays are also its CSC arrays: we pass A.T, a CSC view
    of the same arrays, instead of a converted copy.
    info: number of unknowns, nonzeros in the factors, factorization time."""
    ordering = ordering or ORDERING
    t0 = time.perf_counter()
    if ordering == 'COLAMD':                             # scipy's default (as in spsolve), for comparison
        lu = spla.splu(A.T)
    else:
        lu = spla.splu(A.T, permc_spec=ordering, diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
    t_factor = time.perf_counter() - t0
    x = lu.solve(rhs)
    return x, dict(n_free=A.shape[0], lu_nnz=lu.nnz, t_factor=t_factor)


def solve_with_fixed_nodes(K, fixed, u_fixed, ordering=None):
    """Minimize u^T K u over u with u = u_fixed on the nodes where `fixed` is True.
    Returns (u, info) (see solve_spd)."""
    A, rhs = reduce_to_free(K, fixed, u_fixed)
    u = np.array(u_fixed, dtype=float)
    u[~fixed], info = solve_spd(A, rhs, ordering)
    return u, info


# ---- the IDE quarter cell ------------------------------------------------------------------------
def cell_lines(hc=None, growth=0.12, hx_max=0.5, hy_max=4.0, y_extent_in_a=3.0, h_tp=None,
               h_0=None, y_far=None, s=S_GAP, w=W_FINGER, te=T_E, tp=T_P):
    """Mesh lines (xl, yl) of the quarter cell; see QuarterCell for the parameters."""
    a = s + w
    Y = y_extent_in_a * a
    hc = te / 16 if hc is None else hc
    h_tp = 4 * hc if h_tp is None else h_tp
    h_0 = hc if h_0 is None else h_0
    if not (te >= 0 and te < tp <= Y):
        raise ValueError('need 0 <= t_e < t_p <= Y (the film must cover the electrode)')
    xl = graded_lines([0.0, s / 2, a / 2], [hx_max, hc, hx_max], hx_max, growth)
    # y key lines and element sizes there; a dict merges coinciding keys (t_e = 0 for a thin
    # electrode, t_p = Y for a film filling the upper half); the outer boundaries get the cap
    h_Y = hy_max if y_far is None else hy_max * (1.0 + Y / y_far)
    yk = {tp: h_tp, te: hc, 0.0: hc if te == 0 else h_0}
    yk.update({-Y: h_Y, Y: h_Y})
    yl = graded_lines(sorted(yk), [yk[k] for k in sorted(yk)], hy_max, growth, far=y_far)
    return xl, yl


def count_nodes(p=2, **params):
    """Number of nodes of QuarterCell(p, **params), without building it."""
    xl, yl = cell_lines(**params)
    return (p * (len(xl) - 1) + 1) * (p * (len(yl) - 1) + 1)


class QuarterCell:
    """FEM of the IDE quarter cell on a graded tensor-product mesh.

    p        : element order, 1 (bilinear) or 2 (biquadratic)
    hc       : element size at the electrode side face x = s/2 and top y = t_e (default t_e/16)
    growth   : growth of the element size per unit distance from the nearest key line
    hx_max, hy_max : largest element sizes in x and y
    h_tp     : element size at the film top y = t_p (default 4 hc, as fem2d.py)
    h_0      : element size at the substrate surface y = 0 (default hc, as fem2d.py)
    y_far    : None (fem2d.py's recipe) or a length L: the cap on the y size grows as
               hy_max (1 + |y| / L) (see graded_lines)
    s, w, te, tp, y_extent_in_a : geometry (default: the paper's IDE and film, Y = 3a)
    The defaults reproduce fem2d.py's mesh exactly.  The mesh and the per-region matrices are
    built here; S() then costs one sparse solve.
    """

    def __init__(self, p=2, hc=None, growth=0.12, hx_max=0.5, hy_max=4.0, y_extent_in_a=3.0,
                 h_tp=None, h_0=None, y_far=None, s=S_GAP, w=W_FINGER, te=T_E, tp=T_P):
        hc = te / 16 if hc is None else hc
        self.te = te
        self.params = dict(p=p, hc=hc, growth=growth, hx_max=hx_max, hy_max=hy_max,
                           y_extent_in_a=y_extent_in_a, h_tp=h_tp, h_0=h_0, y_far=y_far)
        xl, yl = cell_lines(hc, growth, hx_max, hy_max, y_extent_in_a, h_tp, h_0, y_far, s, w, te, tp)
        self.mesh = m = TensorMesh(xl, yl, p)
        self.n_nodes = m.n_nodes

        # Key lines are mesh lines; find their indices (exact: graded_lines copies the keys).
        def index(lines, value):
            k = int(np.searchsorted(lines, value))
            assert lines[k] == value
            return k
        Is, J0, Je, Jp = index(xl, s / 2), index(yl, 0.0), index(yl, te), index(yl, tp)

        # Elements by region, from their mesh-line indices.  The electrode elements belong to no
        # region: no field inside the conductor.
        in_gap = m.I < Is
        electrode = ~in_gap & (m.J >= J0) & (m.J < Je)
        self.region = {'sub': m.J < J0,
                       'film': (m.J >= J0) & (m.J < Jp) & ~electrode,
                       'air': m.J >= Jp}
        self.K = {r: m.stiffness(mask) for r, mask in self.region.items()}

        # Nodes: grid indices (i, j); node-grid index of a mesh line = p * line index.
        i, j = m.node_grid()
        i_s, j0, je, last_i, last_j = p * Is, p * J0, p * Je, m.NX - 1, m.NY - 1
        on_electrode = (i >= i_s) & (j >= j0) & (j <= je)              # surface included
        inside_electrode = (i > i_s) & (j > j0) & (j < je)             # surface excluded
        # primal: V = 0 at x = 0 and y = +-Y, V = 1 on the electrode
        self.fixed_V = (i == 0) | (j == 0) | (j == last_j) | on_electrode
        self.V_bc = on_electrode.astype(float)
        # dual: psi = 0 on x = a/2 below the electrode, 1 above it; nodes strictly inside the
        # electrode belong to no element of the energy and are removed (fixed, at 0)
        below = (i == last_i) & (j <= j0)
        above = (i == last_i) & (j >= je)
        self.fixed_psi = inside_electrode | below | above
        self.psi_bc = above.astype(float)
        self.last = {}

    def S(self, eps_film, eps_sub, formulation='primal', ordering=None, release=False):
        """Upper bound on S (formulation 'primal') or lower bound ('dual').  The solution (V or psi
        at the nodes) is kept in self.u, and solver statistics in self.last.
        release=True frees the per-region matrices before the factorization, to save memory in
        one-shot runs on very large meshes (S can then not be called again)."""
        if formulation == 'primal':
            c = {'sub': eps_sub, 'film': eps_film, 'air': 1.0}
            fixed, u_bc = self.fixed_V, self.V_bc
        elif formulation == 'dual':
            if self.te == 0:     # psi jumps across a zero-thickness electrode: not on this mesh
                raise ValueError('the dual formulation needs an electrode of finite thickness')
            c = {'sub': 1.0 / eps_sub, 'film': 1.0 / eps_film, 'air': 1.0}
            fixed, u_bc = self.fixed_psi, self.psi_bc
        else:
            raise ValueError("formulation must be 'primal' or 'dual'")
        if self.K is None:
            raise RuntimeError('the matrices were released by an earlier call with release=True')
        t0 = time.perf_counter()
        K = sum(c[r] * self.K[r] for r in self.K)
        A, rhs = reduce_to_free(K, fixed, u_bc)
        del K                                            # only A is needed from here on
        if release:
            self.K = None
        self.u = np.array(u_bc, dtype=float)
        self.u[~fixed], self.last = solve_spd(A, rhs, ordering)
        del A
        coef = sum(c[r] * self.region[r] for r in self.region)       # coefficient of each element
        W = self.mesh.energy(self.u, coef)
        self.last['t_S'] = time.perf_counter() - t0
        return 4.0 / np.pi * W if formulation == 'primal' else 4.0 / np.pi / W

    def bounds(self, eps_film, eps_sub):
        """(lower, upper) bounds on S from the same mesh."""
        return self.S(eps_film, eps_sub, 'dual'), self.S(eps_film, eps_sub, 'primal')


# ---- a one-parameter family of biquadratic meshes -------------------------------------------------
def p2_mesh(g, te=T_E):
    """QuarterCell parameters for p = 2, all tied to the growth g (convergence study, Nov 2026):

    * corner size hc = (t_e/4096) (g/0.1)^3.  The corner error goes as hc^(4/3) (the potential
      goes as r^(2/3) at the electrode's top corner, inside the film), the grading error roughly
      as g^4, so hc ~ g^3 keeps the two in step; hc smaller than this costs nodes and gains little.
    * y = 0 gets 16 hc: the bottom corner's singularity is weaker (r^0.865: film above, quartz
      below), so it tolerates larger elements; this saves about 9% of the nodes at no cost.
    * caps hx_max = 3.5 g, hy_max = 3 g (um), and the film top 0.3 g: away from the corners the
      potential varies smoothly on scales of a few um (in the gap, the film's in-plane screening
      length eps_film t_p / (eps_sub + 1) = 3.7 um; far above and below, a/pi = 25 um).
    * y_far = 16 um: the y cap grows away from the electrode plane (see graded_lines).
    For g = 0.14, 0.1, 0.08: N = 0.27, 0.59, 0.98 million nodes; bracket width (S_upper/S_lower - 1)
    3.2e-8, 7.5e-9, 3.0e-9, roughly 2.6e3 / N^1.84.
    """
    hc = te / 4096 * (g / 0.1) ** 3
    return dict(p=2, hc=hc, growth=g, hx_max=3.5 * g, hy_max=3.0 * g, h_tp=0.3 * g, h_0=16 * hc,
                y_far=16.0)


# ---- memory ---------------------------------------------------------------------------------------
def peak_memory_mb():
    """Peak resident memory of this process so far, in MB (None if it cannot be found)."""
    try:
        import resource                                  # Linux (kB) and macOS (bytes)
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**20 if sys.platform == 'darwin' else r / 1024
    except ImportError:
        pass
    try:                                                 # Windows: GetProcessMemoryInfo
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in (
                           'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                           'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                           'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        k32 = ctypes.WinDLL('kernel32')
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi')
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 2**20
    except Exception:
        pass
    return None
