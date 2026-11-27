"""
bench_fem2d_cheapest.py -- (d1) how fast can fem2d.py match the model's accuracy if its mesh is tuned?

fem2d.py's mesh has four knobs: the corner element size h_c, the growth rate of the elements away from
the corners, and the largest element sizes hx_max and hy_max.  This script searches them for the mesh
with the shortest solve time whose S is at or below the model's S (every FEM S is an upper bound, so
S <= S_model means "at least as accurate as the model"):

  stage 1  h_c = t_e/n for n in 4..48 and growth 0.04..0.35, with fem2d's default hx_max, hy_max;
  stage 2  the 6 qualifying meshes with the fewest nodes, with hx_max in {0.5, 1, 2} um and
           hy_max in {4, 8, 16} um;
  final    the 5 qualifying meshes with the fewest nodes are re-timed (median of 5) and the fastest
           is logged as 'fem2d.py, tuned mesh'.

The search itself is not timed as part of the result: it is the tuning a user would do once for a
given electrode geometry (and it relies on knowing the model's S).  The same mesh would serve for all
films measured on these electrodes, with somewhat different accuracy.

Run:   python bench_fem2d_cheapest.py          (about 45 s, check included)

Expected (Claude's sandbox, Xeon 2.1 GHz): best mesh h_c = t_e/6, growth 0.06, hx_max 2 um, hy_max 8 um:
    30,997 nodes, S = 7.1889066 (+0.0112%), setup 0.013 s, solve 0.14 s

CONVERGENCE CHECK.  The tuned mesh is then solved once more with every element size halved (h_c, the
growth, hx_max and hy_max; fem2d.py ties the other sizes to h_c), timed like the final re-timing
(setup median of 3, solve median of 5), and logged as 'check: <tuned mesh label>' under the method
'fem2d.py, tuned mesh' (bench_common.record_check).  Claude's sandbox, 21 Nov 2026:
    tuned mesh hmax 2/8 um -> check h_c = t_e/12, growth 0.03, hmax 1/4 um: 122,820 nodes,
    S = 7.1883255143; estimate +8.08e-05 (true +1.12e-04).
    The final choice depends on timings: another run chose hmax 2/16 um (30,024 nodes), whose check
    has 118,956 nodes, S = 7.1883252506, estimate +8.10e-05 (true +1.12e-04); setup + solve 1.0 s
    against 0.17 s for the tuned mesh (x7).
"""
import itertools

import bench_common as bc          # first: pins numpy to one thread

from fem2d import FEM2D
from ide_model import IDEGeometry

G = IDEGeometry(s_um=bc.S_GAP, w_um=bc.W_FINGER, te_um=bc.T_E)


def run(div, growth, hx_max=0.5, hy_max=4.0):
    fem, t_setup = bc.timed(FEM2D, G, bc.T_P, hc_um=bc.T_E / div, growth=growth,
                            hx_max_um=hx_max, hy_max_um=hy_max)
    S, t_solve = bc.timed(fem.S, bc.EPS_FILM, bc.EPS_SUB)
    return dict(div=div, growth=growth, hx_max=hx_max, hy_max=hy_max, n=fem.n_nodes, S=S,
                t=t_setup + t_solve)


def label(r):
    return f"h_c t_e/{r['div']}, growth {r['growth']}, hmax {r['hx_max']:g}/{r['hy_max']:g} um"


def main():
    bc.banner('(d1) fem2d.py with a tuned mesh: cheapest mesh that matches the model')
    print('Stage 1: h_c and growth')
    found = []
    for div, growth in itertools.product([4, 6, 8, 12, 16, 24, 32, 48],
                                         [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.35]):
        r = run(div, growth)
        if r['S'] <= bc.S_MODEL:
            found.append(r)
            print(f"   {label(r):45s} {r['n']:8,} nodes  S = {r['S']:.7f}  ({r['t']:.2f} s)", flush=True)
    if not found:
        print('No mesh in stage 1 reached the model accuracy.')
        return
    print('Stage 2: largest element sizes, for the 6 smallest qualifying meshes')
    for r0 in sorted(found, key=lambda r: r['n'])[:6]:
        for hx_max, hy_max in itertools.product([0.5, 1.0, 2.0], [4.0, 8.0, 16.0]):
            if (hx_max, hy_max) == (0.5, 4.0):
                continue
            r = run(r0['div'], r0['growth'], hx_max, hy_max)
            if r['S'] <= bc.S_MODEL:
                found.append(r)
                print(f"   {label(r):45s} {r['n']:8,} nodes  S = {r['S']:.7f}  ({r['t']:.2f} s)", flush=True)
    print('Final: re-time the 5 smallest qualifying meshes (median of 5 solves)')
    best = None
    for r in sorted(found, key=lambda r: r['n'])[:5]:
        fem, t_setup = bc.median_time(lambda: FEM2D(G, bc.T_P, hc_um=bc.T_E / r['div'], growth=r['growth'],
                                                    hx_max_um=r['hx_max'], hy_max_um=r['hy_max']), 3)
        S, t_solve = bc.median_time(lambda: fem.S(bc.EPS_FILM, bc.EPS_SUB), 5)
        print(f"   {label(r):45s} {fem.n_nodes:8,} nodes  setup {t_setup:.3f} s  solve {t_solve:.3f} s", flush=True)
        if best is None or t_setup + t_solve < best[2] + best[3]:
            best = (r, fem.n_nodes, t_setup, t_solve, S)
    r, n, t_setup, t_solve, S = best
    bc.record('fem2d.py, tuned mesh', label(r), n, t_setup, t_solve, S, per_eval_s=t_solve,
              notes='cheapest mesh with S <= S_model found by the scan')

    # convergence check: the tuned mesh with every element size halved, timed like the re-timing above
    q = dict(hc_um=bc.T_E / r['div'] / 2, growth=r['growth'] / 2, hx_max_um=r['hx_max'] / 2,
             hy_max_um=r['hy_max'] / 2)
    fem, t_setup_c = bc.median_time(lambda: FEM2D(G, bc.T_P, **q), 3)
    S_c, t_solve_c = bc.median_time(lambda: fem.S(bc.EPS_FILM, bc.EPS_SUB), 5)
    bc.record_check('fem2d.py, tuned mesh', label(r), S, fem.n_nodes, t_setup_c, t_solve_c, S_c,
                    extra_notes=f"h_c = t_e/{2 * r['div']:g}, growth {q['growth']:g}, "
                                f"hmax {q['hx_max_um']:g}/{q['hy_max_um']:g} um")


if __name__ == '__main__':
    main()
