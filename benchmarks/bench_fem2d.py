"""
bench_fem2d.py -- (b3) the finite-element code used in the paper, fem2d.py (bilinear rectangles on a
mesh graded toward the electrode corners, sparse direct solve with SuperLU), on the paper's mesh
ladder (Supplemental Section S10) plus a finer mesh.

What is timed
-------------
  setup   FEM2D(...): mesh lines and the per-region stiffness matrices.
  solve   fem.S(eps_film, eps_sub): K(eps) = sum of eps_r K_r, factor and solve (SuperLU), energy.
          Each further S with new dielectric constants needs one more solve on the same mesh, so
          per_eval_s = solve time.

Every S is an upper bound on the true value (Dirichlet's principle) and decreases as the mesh is
refined; "S <= S_model" therefore means "at least as accurate as the model".

Run:   python bench_fem2d.py          (about 1 min, check included)

Expected (Claude's sandbox, Xeon 2.1 GHz; your times will be shorter):
    h_c = t_e/8,   growth 0.20:  18,725 nodes  S = 7.1912802  (+0.0442%)  solve 0.08 s
    h_c = t_e/16,  growth 0.12:  30,375 nodes  S = 7.1897879  (+0.0234%)  solve 0.14 s
    h_c = t_e/24,  growth 0.08:  49,824 nodes  S = 7.1890143  (+0.0127%)  solve 0.28 s
    h_c = t_e/32,  growth 0.06:  75,402 nodes  S = 7.1886607  (+0.0078%)  solve 0.50 s
    h_c = t_e/48,  growth 0.05: 109,056 nodes  S = 7.1884994  (+0.0055%)  solve 0.90 s
    h_c = t_e/64,  growth 0.04: 162,447 nodes  S = 7.1883654  (+0.0037%)  solve 1.3 s  (the paper's finest)
    h_c = t_e/128, growth 0.02: 598,502 nodes  S = 7.1881700  (+0.0009%)  solve 8.1 s

CONVERGENCE CHECK.  After the ladder, the rung with the fewest nodes that beats the model (h_c = t_e/32,
growth 0.06) is solved once more with every element size halved -- h_c, the growth, and fem2d's caps
hx_max = 0.5 um and hy_max = 4 um (fem2d.py ties the sizes at the substrate surface and the film top to
h_c) -- timed in the same way and logged as 'check: <rung label>' (bench_common.record_check).
Claude's sandbox, 21 Nov 2026 (shared with another job, so times are rough):
    check of t_e/32: h_c = t_e/64, growth 0.03, caps 0.25/2 um: 299,625 nodes, S = 7.1882420714;
    estimate +5.82e-05 (true +7.77e-05); setup + solve 3.5 to 5 s against 0.53 s for the rung
    (x7 to x10: fem2d's solve time grows faster than the number of nodes).
"""
import bench_common as bc          # first: pins numpy to one thread

import inspect

from fem2d import FEM2D
from ide_model import IDEGeometry

RUNGS = [(8, 0.20), (16, 0.12), (24, 0.08), (32, 0.06), (48, 0.05), (64, 0.04), (128, 0.02)]
# fem2d's default caps (used by every rung above); the check halves them too
_DEF = inspect.signature(FEM2D.__init__).parameters
HX_MAX, HY_MAX = _DEF['hx_max_um'].default, _DEF['hy_max_um'].default


def run(g, hc, growth, hx_max=HX_MAX, hy_max=HY_MAX):
    """Setup (timed once) and solve (median of 3 below 200,000 nodes); returns (fem, S, setup, solve)."""
    fem, t_setup = bc.timed(FEM2D, g, bc.T_P, hc_um=hc, growth=growth, hx_max_um=hx_max, hy_max_um=hy_max)
    repeat = 3 if fem.n_nodes < 200000 else 1
    S, t_solve = bc.median_time(lambda: fem.S(bc.EPS_FILM, bc.EPS_SUB), repeat)
    return fem, S, t_setup, t_solve


def main():
    bc.banner('(b3) fem2d.py -- bilinear finite elements, sparse direct solve (the paper\'s FEM)')
    g = IDEGeometry(s_um=bc.S_GAP, w_um=bc.W_FINGER, te_um=bc.T_E)
    done = []
    for div, growth in RUNGS:
        fem, S, t_setup, t_solve = run(g, bc.T_E / div, growth)
        label = f'h_c = t_e/{div}, growth {growth}'
        bc.record('fem2d.py (bilinear, direct)', label, fem.n_nodes, t_setup, t_solve, S, per_eval_s=t_solve)
        done.append(dict(label=label, div=div, growth=growth, n=fem.n_nodes, S=S))
        del fem

    # convergence check of the cheapest rung that beats the model (see the module docstring)
    base = bc.fewest_nodes(done, lambda r: bc.as_accurate_as_model(r['S']))
    if base is None:
        print('No rung reached the model accuracy: no convergence check.')
        return
    hc, growth, hx, hy = bc.T_E / base['div'] / 2, base['growth'] / 2, HX_MAX / 2, HY_MAX / 2
    fem, S, t_setup, t_solve = run(g, hc, growth, hx, hy)
    bc.record_check('fem2d.py (bilinear, direct)', base['label'], base['S'], fem.n_nodes, t_setup, t_solve, S,
                    extra_notes=f"h_c = t_e/{2 * base['div']:g}, growth {growth:g}, hx_max {hx:g} um, hy_max {hy:g} um")


if __name__ == '__main__':
    main()
