"""
check_model.py
==============

Quick checks of ide_model.py (a few seconds).  Run it after installing, or after changing the code:

    python check_model.py

1. Infinitely thin electrodes between two uniform media: the exact capacitance is known from
   conformal mapping (Sec. 3.2.6, first test).  The optimized in-plane potential should agree to
   about 1 part in 10^5; den Otter's fixed potential (Eq. 3) gives a value about 0.22% too high.
2. Numerical convergence: halving the node spacings of f(x) or doubling the number of Fourier
   terms should change the capacitance of a film-covered IDE by much less than 0.01%.
3. Round trip: computing a capacitance for known dielectric constants and then solving for them
   should return the same values.
"""

import numpy as np

from ide_model import IDEGeometry, IDEModel, conformal_mapping_S, den_otter_S


def main():
    geom = IDEGeometry()                       # the electrodes of the paper
    print('Electrodes: s = %.2f um, w = %.2f um, t_e = %.3f um, N = %d, L = %.3f mm, N_eff = %.3f'
          % (geom.s_um, geom.w_um, geom.te_um, geom.N, geom.L_mm, geom.n_eff()))

    # 1. conformal mapping (t_e = 0, quartz below, air above)
    thin = IDEModel(IDEGeometry(te_um=0.0))
    exact = conformal_mapping_S(geom.s_um, geom.w_um, 1.0, 4.55)
    opt = thin.S_blank(4.55)
    otter = den_otter_S(thin, 1.0, 4.55, 0.0)
    print('\n1. Thin electrodes, exact (conformal mapping) vs model:')
    print('   optimized f(x): %+.2e (relative)   den Otter fixed f(x): %+.3f%%' % (opt / exact - 1, 100 * (otter / exact - 1)))

    # 2. convergence for a MAPbI3-like film (eps = 47, 438 nm) on quartz
    base = IDEModel(geom)
    fine = IDEModel(geom, n_terms=40000, h_edge_um=geom.te_um / 16, h_max_um=1.0)
    s0, s1 = base.S(47.0, 4.52, 0.438), fine.S(47.0, 4.52, 0.438)
    print('\n2. Finer nodes and twice the Fourier terms change S by %+.1e (relative)' % (s1 / s0 - 1))

    # 3. round trip
    C_blank = base.blank_capacitance(4.52)
    C_film = base.capacitance(47.0, 4.52, 0.438)
    q = base.solve_substrate(C_blank)
    e = base.solve_film(C_film, q, 0.438)
    print('\n3. Round trip: eps_Q 4.52 -> %.6f, eps_P 47 -> %.6f   (C_blank = %.3f pF, C_film = %.3f pF)'
          % (q, e, C_blank * 1e12, C_film * 1e12))
    z = base.solve_film_complex(np.real(base.capacitance(47.0 - 5.0j, 4.52, 0.438)),
                                -2 * np.pi * 1e5 * np.imag(base.capacitance(47.0 - 5.0j, 4.52, 0.438)),
                                1e5, 4.52 + 0j, 0.438)
    print('   complex: eps_P 47 - 5i -> %.6f %+.6fi' % (z.real, z.imag))


if __name__ == '__main__':
    main()
