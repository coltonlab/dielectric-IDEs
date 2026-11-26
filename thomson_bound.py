"""
thomson_bound.py
================

A rigorous LOWER bound on the capacitance of the IDE from Thomson's principle, and the checks of
the paper that use it (Sec. 3.2.6 and Supplemental Section S6 of McClure et al. 2026).

    python thomson_bound.py --paper      (every number of Sec. 3.2.6 and Section S6; a few seconds)
    python thomson_bound.py --eps-film 47 --eps-sub 4.52 --thickness-nm 437.8
                                         (one case: the bound, our model at t_e = 0, and their ratio)


WHY A LOWER BOUND
-----------------
ide_model.py chooses, among its trial potentials, the one that minimizes the stored energy.  By
Dirichlet's principle its capacitance is an UPPER bound on the true capacitance (Sec. 3.2.5).
Thomson's principle gives a LOWER bound: of all displacement fields D that have zero divergence
between the conductors and carry the correct charge Q on each conductor, the true field stores the
least energy U, so any such trial field gives C >= Q^2 / (2 U).

The paper uses two inequalities (Sec. 3.2.6):
    C(our electrodes) >= C(infinitely thin electrodes, t_e = 0) >= C_Thomson(t_e = 0).
The first holds because every potential that is admissible for thick electrodes is also admissible
for thin ones.  The second is Thomson's principle, applied to the thin-electrode geometry, where the
trial field can be computed exactly.  This script computes C_Thomson(t_e = 0) and compares it with
our model evaluated at t_e = 0 (an upper bound on the same capacitance).  The two agree to within
1.5 parts in 10^5 for every film and frequency of the paper, so the paper uses our model at t_e = 0
as the lower bound.


THE TRIAL FIELD (Section S6)
----------------------------
Geometry of Fig. 3 with t_e = 0: substrate (eps_Q) below the plane of the electrodes, the film
(eps_P) from that plane up to y = t_p, air above.  On each finger the trial charge per unit area is

    sigma(u) = sum_{j=0}^{J-1} c_j T_2j(u) / sqrt(1 - u^2)                                  (S7)

with the opposite sign on alternate fingers; u is the position across the finger, measured from
its center in units of its half-width w/2, and T_2j are Chebyshev polynomials.  The paper uses
J = 10 terms.  Only the j = 0 term carries charge (the integral of T_2j(u)/sqrt(1 - u^2) across
the finger vanishes for j >= 1), so c_0 = 1 fixes the charge and c_1 ... c_{J-1} are free.
The field of this charge is computed exactly, one Fourier mode at a time, with the same layered
solutions of Laplace's equation as in the model: mode n sees eps_Q below and the film-plus-air
stack above, which acts as a medium of dielectric constant eps_eff,n (IDEModel.eps_above, with
d = t_p).  So the field has zero divergence outside the electrodes, the right normal component at
every interface, and flux Q out of each electrode: it satisfies Thomson's conditions by
construction.  (Its potential is not exactly constant across each finger, which is why it is
only a bound.)  Its energy is a quadratic form in the coefficients:

    C_Thomson = eps0 N L pi / (8 sum_{i,j} c_i c_j M_ij)                                    (S8)
    M_ij = sum_{n>=1} (-1)^(i+j) J_2i(k_n w/2) J_2j(k_n w/2) / ((2n - 1)(eps_Q + eps_eff,n))  (S9)

with k_n = (2n - 1) pi / a (Eq. 14).  (The Fourier coefficients of (S7) are Bessel functions
because the integral of T_2j(u) cos(beta u) / sqrt(1 - u^2) from u = -1 to 1 is
(-1)^j pi J_2j(beta).)  We choose c_1 ... c_{J-1} to make the double sum as small as possible
(J - 1 linear equations), which gives the highest bound of this form.

In the dimensionless units of ide_model.py, C = (eps0 N_eff L pi / 8) S, so S_Thomson = 1 / (c^T M c).
(N_eff, the effective number of half-periods, multiplies the model and the bound alike.)

The sum over n converges only as 1/n^2.  We sum the first 2 x 10^5 terms and add the asymptotic
remainder: for large k_n, (-1)^(i+j) J_2i J_2j -> (2 / (pi beta)) cos^2(beta - pi/4), which averages
to 1 / (pi beta) with beta = k_n w / 2, and eps_eff,n -> eps_P.  The remainder is a few parts in 10^7
of the total.  As in ide_model.py, only the first few hundred modes feel the top of the film; beyond them
eps_eff,n equals eps_P to double precision, so the sum over those modes is computed once per
electrode geometry and reused for every film and frequency.


WHAT --paper PRINTS (each next to the value printed in the paper)
-----------------------------------------------------------------
(a) One medium on each side of the electrodes: the bound against the exact conformal-mapping
    capacitance of Sec. 4.1, with the terms j = 0 and 1 only (Section S6: within 1 part in 10^7).
(b) Our model at t_e = 0 against the ten-term bound, S_model / S_Thomson - 1, for every film run
    at every frequency and for the two blank substrates (Sec. 3.2.6 and Section S6: at most
    1.5 parts in 10^5).  The dielectric constants are those of results/*.csv (reproduce_paper.py).
(c) Fourteen terms instead of ten (Section S6: the bound changes by less than 3 parts in 10^7).
(d) Our model at t_e = 0 with every node spacing of f(x) divided by four: the difference from the
    bound shrinks several-fold (Section S6: most of the remaining difference comes from the
    discretization of f(x) in our model).
(e) At 9.67 MHz, how far the lower bound (our model with t_e = 0) lies below the upper bound (our
    model with t_e = 115 nm) (Sec. 3.2.6: 0.25% for the blank substrate, 0.28% to 0.43% for the
    perovskite films).  The paper's numbers are S_upper / S_lower - 1; 1 - S_lower / S_upper is
    printed in brackets (it differs in the third digit).
(f) The range of eps_P at 9.67 MHz that the two bounds alone allow, given the measured blank and
    film capacitances (Sec. 3.2.6: a few percent for most films, about 10% for PEA2PbCl4).

Requires the files written by reproduce_paper.py (results/*.csv, included in the repository).
"""

import argparse
import csv
import os

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, polygamma

from ide_model import IDEGeometry, IDEModel, PAPER_MAIN_BATCH, PAPER_SECOND_BATCH, conformal_mapping_S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
RESULTS = os.path.join(HERE, 'results')


# --------------------------------------------------------------------------------------------
# The bound
# --------------------------------------------------------------------------------------------
class ThomsonBound:
    """Thomson lower bound S_Thomson for infinitely thin electrodes (t_e = 0) of gap s and width w.

    Create one per electrode geometry (it precomputes the Bessel functions, about a second) and
    call S(eps_film, eps_sub, t_film_um) for each case.

    max_terms : largest number of Chebyshev terms J that will be used (the paper uses 10)
    n_modes   : Fourier modes summed explicitly before the asymptotic remainder (paper: 2 x 10^5)
    """

    E_CUT = 1e-17      # a mode with exp(-2 k_n t_p) below this sees an infinitely thick film

    def __init__(self, s_um: float, w_um: float, max_terms: int = 14, n_modes: int = 200000):
        self.s, self.w, self.a = s_um, w_um, s_um + w_um
        self.max_terms, self.n_modes = max_terms, n_modes
        n = np.arange(1, n_modes + 1)
        self.m = 2.0 * n - 1.0                      # 2n - 1
        self.k = self.m * np.pi / self.a            # k_n (1/um)
        beta = self.k * self.w / 2.0
        # P[j, n] = (-1)^j J_2j(k_n w/2), so that (-1)^(i+j) J_2i J_2j = P[i, n] P[j, n]
        self.P = np.array([(-1.0) ** j * jv(2 * j, beta) for j in range(max_terms)])
        # sum over ALL explicit modes with weight 1 / (2n - 1); for each film only the first modes
        # need their own weight (see S below)
        self.G_all = (self.P / self.m) @ self.P.T
        self._head = {}                             # cache of partial sums over the first n modes
        # asymptotic remainder per unit 1/eps_sum:
        # sum_{n > n_modes} 2a / (pi^2 w (2n - 1)^2),  with  sum_{n > N} 1/(2n - 1)^2 = psi_1(N + 1/2) / 4
        self.tail_unit = 2.0 * self.a / (np.pi ** 2 * self.w) * polygamma(1, n_modes + 0.5) / 4.0
        # the model's layered response of the film plus air (IDEModel.eps_above), for t_e = 0
        self._layer = IDEModel(IDEGeometry(s_um=s_um, w_um=w_um, te_um=0.0))

    def _n_film_modes(self, t_film_um):
        """Number of leading modes whose weight depends on the film thickness (cf. IDEModel)."""
        if t_film_um <= 0:
            return 0
        n1 = int(np.floor((np.log(1.0 / self.E_CUT) * self.a / (2.0 * np.pi * t_film_um) + 1.0) / 2.0))
        return max(0, min(n1, self.n_modes))

    def M(self, eps_film, eps_sub, t_film_um):
        """The matrix M_ij of Eq. (S9), i, j = 0 ... max_terms - 1 (real dielectric constants only)."""
        if np.iscomplexobj(eps_film) or np.iscomplexobj(eps_sub):
            raise ValueError('The Thomson bound is defined for real dielectric constants only.')
        if t_film_um <= 0:
            # no film: every mode sees air above, so the weight is the same for all of them
            e_inf = eps_sub + 1.0
            return self.G_all / e_inf + self.tail_unit / e_inf
        e_inf = eps_sub + eps_film                  # weight of the modes that see a thick film
        n1 = self._n_film_modes(t_film_um)
        if n1 not in self._head:
            P1 = self.P[:, :n1]
            self._head[n1] = (P1 / self.m[:n1]) @ P1.T
        P1, m1, k1 = self.P[:, :n1], self.m[:n1], self.k[:n1]
        es = eps_sub + self._layer.eps_above(eps_film, t_film_um, k=k1)     # eps_Q + eps_eff,n
        head = (P1 / (m1 * es)) @ P1.T
        return head + (self.G_all - self._head[n1]) / e_inf + self.tail_unit / e_inf

    def S(self, eps_film, eps_sub, t_film_um, terms: int = 10, return_coefficients=False):
        """Dimensionless lower bound S_Thomson (C >= eps0 N_eff L pi/8 * S_Thomson) with `terms`
        Chebyshev terms, for a film of thickness t_film_um (from the plane of the electrodes) and
        dielectric constant eps_film on a substrate eps_sub.  Blank substrate: eps_film = 1, t = 0."""
        if not 1 <= terms <= self.max_terms:
            raise ValueError('terms must be between 1 and max_terms = %d' % self.max_terms)
        Mm = self.M(eps_film, eps_sub, t_film_um)[:terms, :terms]
        c = np.ones(1)
        if terms > 1:                               # c_0 = 1; minimize c^T M c over c_1 ... c_{J-1}
            c = np.concatenate([[1.0], -np.linalg.solve(Mm[1:, 1:], Mm[1:, 0])])
        S = 1.0 / (c @ Mm @ c)
        return (S, c) if return_coefficients else S


# --------------------------------------------------------------------------------------------
# The checks of the paper
# --------------------------------------------------------------------------------------------
def _rows(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith('#')))


def _nearest(f, target):
    return int(np.argmin(np.abs(np.log(np.asarray(f) / target))))


def paper():
    samples = [r for r in _rows(os.path.join(DATA, 'samples.csv')) if r['compound'] != 'water']
    geoms = {'main': PAPER_MAIN_BATCH, 'second': PAPER_SECOND_BATCH}
    te = PAPER_MAIN_BATCH.te_um
    thin = {b: IDEGeometry(s_um=g.s_um, w_um=g.w_um, te_um=0.0) for b, g in geoms.items()}
    print('Setting up the bound for the two wafer batches ...', flush=True)
    bound = {b: ThomsonBound(g.s_um, g.w_um) for b, g in geoms.items()}
    model0 = {b: IDEModel(g) for b, g in thin.items()}                     # our model at t_e = 0
    model = {b: IDEModel(g) for b, g in geoms.items()}                     # our model at t_e = 115 nm
    m0 = model0['main']
    # every node spacing of f(x) divided by four (defaults at t_e = 0: h_edge = s/30000,
    # h_max = min(2 um, s/25), growth 0.15)
    model0_fine = {b: IDEModel(g, h_edge_um=g.s_um / 30000 / 4, h_max_um=min(2.0, g.s_um / 25) / 4,
                               growth=0.15 / 4) for b, g in thin.items()}

    # (a) one medium on each side: exact conformal-mapping capacitance
    g = PAPER_MAIN_BATCH
    exact = conformal_mapping_S(g.s_um, g.w_um, 1.0, 1.0)
    print('\n(a) One medium on each side of the electrodes, S_Thomson / conformal mapping - 1')
    print('    (Section S6: within 1 part in 10^7 with the terms j = 0 and 1)')
    for J in (1, 2, 3, 10):
        print(f'    terms j = 0 ... {J - 1}: {bound["main"].S(1.0, 1.0, 0.0, terms=J) / exact - 1:+.1e}')

    # (b)-(d) every film run and frequency, and the two blanks
    print('\n(b) Our model at t_e = 0 against the ten-term bound, S_model / S_Thomson - 1')
    print('    (Sec. 3.2.6 and Section S6: at most 1.5 parts in 10^5 for all samples and frequencies)')
    gap10, gap14, gapfine = [], [], []
    worst = (-1.0, '')
    cases = [(f'blank ({b} batch)', b, 'quartz_%s_batch.csv' % b, None) for b in geoms]
    cases += [(r['label'], r['batch'], r['label'].replace(' ', '_') + '.csv', float(r['thickness_nm']) / 1000.0)
              for r in samples]
    for label, b, fname, tp in cases:
        rows = _rows(os.path.join(RESULTS, fname))
        g10, g14, gf = [], [], []
        for d in rows:
            q = float(d['eps_substrate'])
            e, t = (1.0, 0.0) if tp is None else (float(d['eps_r']), tp)
            s10 = bound[b].S(e, q, t, terms=10)
            s14 = bound[b].S(e, q, t, terms=14)
            g10.append(model0[b].S(e, q, t) / s10 - 1.0)
            g14.append(s14 / s10 - 1.0)
            gf.append(model0_fine[b].S(e, q, t) / s10 - 1.0)
            if g10[-1] > worst[0]:
                worst = (g10[-1], f'{label} at {float(d["f_Hz"]):.4g} Hz')
        gap10 += g10; gap14 += g14; gapfine += gf
        print(f'    {label:20s} {min(g10):+.2e} .. {max(g10):+.2e}   ({len(rows)} frequencies)', flush=True)
    gap10, gap14, gapfine = map(np.array, (gap10, gap14, gapfine))
    print(f'    all: {gap10.min():+.2e} .. {gap10.max():+.2e}, largest for {worst[1]}; '
          f'below zero (which would contradict the bound): {int((gap10 < 0).sum())} points')
    print('\n(c) Fourteen terms instead of ten: the bound changes by '
          f'{gap14.min():+.1e} .. {gap14.max():+.1e}   (Section S6: less than 3 parts in 10^7)')
    print(f'\n(d) Our model at t_e = 0 with every node spacing of f(x) divided by four ({len(model0_fine["main"].x)} '
          f'nodes instead of {len(m0.x)}): S_model / S_Thomson - 1 = {gapfine.min():+.1e} .. {gapfine.max():+.1e}')
    print('    (Section S6: most of the remaining difference in (b) comes from the discretization of f(x))')

    # (e) and (f) at 9.67 MHz
    print('\n(e) At 9.67 MHz: how far the lower bound (our model, t_e = 0) lies below the upper bound (our model,')
    print('    t_e = 115 nm), as S_upper / S_lower - 1 (in brackets: 1 - S_lower / S_upper)')
    print('    (Sec. 3.2.6: 0.25% for the blank substrate, 0.28% to 0.43% for the films)')
    q9 = {}
    for b in geoms:
        rows = _rows(os.path.join(RESULTS, 'quartz_%s_batch.csv' % b))
        j = _nearest([float(r['f_Hz']) for r in rows], 9.67e6)
        q = q9[b] = float(rows[j]['eps_substrate'])
        up, lo = model[b].S(1.0, q, te), model0[b].S(1.0, q, 0.0)
        print(f'    blank ({b} batch), eps_Q = {q:.3f}: {100 * (up / lo - 1):.3f}%  ({100 * (1 - lo / up):.3f}%)')
    film_gaps, brackets = [], []
    for r in samples:
        b, tp = r['batch'], float(r['thickness_nm']) / 1000.0
        rows = _rows(os.path.join(RESULTS, r['label'].replace(' ', '_') + '.csv'))
        j = _nearest([float(x['f_Hz']) for x in rows], 9.67e6)
        q, e = float(rows[j]['eps_substrate']), float(rows[j]['eps_r'])   # Cp alone at 9.67 MHz
        up, lo = model[b].S(e, q, tp), model0[b].S(e, q, tp)
        film_gaps.append(up / lo - 1)
        # (f) The measured capacitances are those our model gives for the extracted (q, e).  The true
        # capacitance lies between the two bounds, for the blank and for the film.  The largest eps_Q
        # consistent with the blank is where the LOWER bound reaches the measured blank capacitance;
        # the smallest eps_P is then where the UPPER bound, with that eps_Q, reaches the measured film
        # capacitance; the largest eps_P is where the LOWER bound, with the smallest eps_Q (the
        # extracted one, from the upper bound), reaches it.
        Sb, Sf = model[b].S(1.0, q, te), up
        q_hi = brentq(lambda x: model0[b].S(1.0, x, 0.0) - Sb, q, 1.5 * q, xtol=1e-12)
        e_lo = brentq(lambda x: model[b].S(x, q_hi, tp) - Sf, 1.0001, e, xtol=1e-10)
        e_hi = brentq(lambda x: model0[b].S(x, q, tp) - Sf, e, 2.0 * e, xtol=1e-10)
        brackets.append((r['label'], r['compound'], e, 100 * (e_lo / e - 1), 100 * (e_hi / e - 1)))
        print(f'    {r["label"]:12s} eps_r = {e:6.2f}: {100 * film_gaps[-1]:.3f}%  ({100 * (1 - lo / up):.3f}%)')
    print(f'    films: {100 * min(film_gaps):.2f}% to {100 * max(film_gaps):.2f}%')
    print('\n(f) Range of eps_P at 9.67 MHz allowed by the two bounds alone, in percent of the extracted eps_r')
    print('    (Sec. 3.2.6: a few percent for most films, about 10% for PEA2PbCl4)')
    for label, comp, e, lo_pct, hi_pct in brackets:
        print(f'    {label:12s} {comp:10s} eps_r = {e:6.2f}:  {lo_pct:+5.1f}% .. {hi_pct:+5.1f}%   (width {hi_pct - lo_pct:4.1f}%)')


def main(argv=None):
    p = argparse.ArgumentParser(description="Thomson lower bound on the IDE capacitance (Sec. 3.2.6 and "
                                            "Supplemental Section S6 of McClure et al. 2026).")
    p.add_argument('--paper', action='store_true', help='reproduce the numbers of Sec. 3.2.6 and Section S6')
    p.add_argument('--eps-film', type=float, default=47.0, help='film dielectric constant (default 47)')
    p.add_argument('--eps-sub', type=float, default=4.52, help='substrate dielectric constant (default 4.52)')
    p.add_argument('--thickness-nm', type=float, default=437.8, help='film thickness, nm (0 for a blank substrate)')
    p.add_argument('--s-um', type=float, default=50.27, help='gap between electrodes, um (default 50.27)')
    p.add_argument('--w-um', type=float, default=29.78, help='electrode width, um (default 29.78)')
    p.add_argument('--terms', type=int, default=10, help='number of Chebyshev terms (default 10, at most 14)')
    a = p.parse_args(argv)
    if a.paper:
        paper()
        return
    t = a.thickness_nm / 1000.0
    e = a.eps_film if t > 0 else 1.0
    T = ThomsonBound(a.s_um, a.w_um, max_terms=max(14, a.terms))
    S_lb, c = T.S(e, a.eps_sub, t, terms=a.terms, return_coefficients=True)
    S_m = IDEModel(IDEGeometry(s_um=a.s_um, w_um=a.w_um, te_um=0.0)).S(e, a.eps_sub, t)
    print(f'Thomson lower bound S = {S_lb:.8f} ({a.terms} terms), our model at t_e = 0: S = {S_m:.8f}, '
          f'ratio - 1 = {S_m / S_lb - 1:+.2e}')
    print('coefficients c_j:', ' '.join(f'{x:+.4e}' for x in c))


if __name__ == '__main__':
    main()
