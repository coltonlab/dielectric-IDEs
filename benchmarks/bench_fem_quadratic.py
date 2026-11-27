"""
bench_fem_quadratic.py (d2) -- biquadratic finite elements (fem_tensor.py, p = 2, primal): time and
accuracy on a ladder of meshes from coarse to far beyond the model's accuracy.

Each rung: build the mesh and the per-region matrices (setup), then solve for S (an UPPER bound on
the true S; Dirichlet's principle).  Solving again for other dielectric constants reuses the setup,
so per_eval_s = the solve time.  Solve and setup times are medians of several runs when they are
cheap (21 runs below 0.05 s, 7 below 0.2 s, 3 below 5 s), because the timings are noisy.

The rungs: six coarse meshes, the cheapest ones found (Nov 2026 study) for errors around the
model's (+1.1e-4): large growth (0.6 to 0.7), large caps, a few elements across the electrode;
then the mesh family fem_tensor.p2_mesh(g) used for the reference bracket (reference_bounds.py).

    python bench_fem_quadratic.py          (about 30 s here)

Expected output (Claude's sandbox, Intel Xeon 2.1 GHz shared with other jobs, 21 Nov 2026).  The
error column is relative to bench_common.S_REF: shown here for S_REF = 7.18810231557 (bracket
width 1.4e-9, reference_bounds.py); with the older provisional S_REF = 7.1880977 it reads 6.4e-7
higher.  Millisecond timings vary by up to 2x between runs on a busy machine.
    rung                                       nodes            S  S/S_REF-1   setup s   solve s  beats model
    coarse: g 0.7, hc t_e/4, caps 6/6          2,035  7.190209348  +2.93e-04    0.0025    0.0043  no
    coarse: g 0.7, hc t_e/4, caps 6/4          2,479  7.189474091  +1.91e-04    0.0026    0.0051  no
    coarse: g 0.6, hc t_e/4, caps 6/4          2,993  7.188888026  +1.09e-04    0.0028    0.0055  yes
    coarse: g 0.6, hc t_e/6, caps 6/4          3,465  7.188548693  +6.21e-05    0.0030    0.0094  yes
    coarse: g 0.6, hc t_e/8, caps 4/4          5,151  7.188342371  +3.34e-05    0.0048    0.0134  yes
    coarse: g 0.6, hc t_e/12, caps 2/4         7,383  7.188240646  +1.92e-05    0.0059    0.0205  yes
    p2_mesh(g = 0.5)                          14,007  7.188154019  +7.19e-06    0.0075    0.0310  yes
    p2_mesh(g = 0.4)                          24,035  7.188112483  +1.41e-06    0.0118    0.0708  yes
    p2_mesh(g = 0.3)                          46,851  7.188105372  +4.25e-07    0.0212    0.1607  yes
    p2_mesh(g = 0.2)                         118,991  7.188102817  +6.97e-08    0.0577    0.5210  yes
    p2_mesh(g = 0.14)                        269,885  7.188102429  +1.57e-08    0.1378    1.4362  yes
    p2_mesh(g = 0.1)                         586,639  7.188102342  +3.68e-09    0.3781    3.7442  yes
The model (ide_model.py, +1.14e-4) takes about 4 ms per S.  The cheapest mesh here that beats it
(2,993 nodes) takes about 5.5 ms per S; most of that is SuperLU's factorization (4.4 ms).  Caveat:
the coarse meshes were picked by comparing with the reference value for THIS geometry; without a
reference one would need a convergence study (or the lower bound) to know the error.

CONVERGENCE CHECK.  After the ladder, the rung with the fewest nodes that beats the model is solved
once more on a mesh with every element size halved (h_c, h_0, h_tp, the caps and the growth; y_far
unchanged), timed in the same way, and logged as 'check: <rung label>' (bench_common.record_check).
S_rung/S_check - 1 is what a user would take as the rung's error.  Claude's sandbox, 21 Nov 2026:
    check of the 2,993-node rung: 12,495 nodes, S = 7.1881583182; estimate +1.02e-04 (true +1.09e-04);
    setup + solve 0.039 s against 0.009 s for the rung, so first S + check is about 5x the first S.
"""
import bench_common as bc       # first: pins numpy/scipy to one thread

import numpy as np

import fem_tensor as ft

GEOM = dict(s=bc.S_GAP, w=bc.W_FINGER, te=bc.T_E, tp=bc.T_P, y_extent_in_a=bc.Y_EXTENT_IN_A)
te = bc.T_E

# (label, QuarterCell parameters); coarse meshes first
RUNGS = [
    ('coarse: g 0.7, hc t_e/4, caps 6/6', dict(p=2, growth=0.7, hc=te / 4, h_0=te, h_tp=0.2,
                                               hx_max=6.0, hy_max=6.0, y_far=16.0)),
    ('coarse: g 0.7, hc t_e/4, caps 6/4', dict(p=2, growth=0.7, hc=te / 4, h_0=te, h_tp=0.2,
                                               hx_max=6.0, hy_max=4.0, y_far=16.0)),
    ('coarse: g 0.6, hc t_e/4, caps 6/4', dict(p=2, growth=0.6, hc=te / 4, h_0=te, h_tp=0.2,
                                               hx_max=6.0, hy_max=4.0, y_far=16.0)),
    ('coarse: g 0.6, hc t_e/6, caps 6/4', dict(p=2, growth=0.6, hc=te / 6, h_0=4 * te / 6, h_tp=0.2,
                                               hx_max=6.0, hy_max=4.0, y_far=16.0)),
    ('coarse: g 0.6, hc t_e/8, caps 4/4', dict(p=2, growth=0.6, hc=te / 8, h_0=te / 2, h_tp=0.2,
                                               hx_max=4.0, hy_max=4.0, y_far=32.0)),
    ('coarse: g 0.6, hc t_e/12, caps 2/4', dict(p=2, growth=0.6, hc=te / 12, h_0=te / 3, h_tp=0.2,
                                                hx_max=2.0, hy_max=4.0, y_far=32.0)),
] + [(f'p2_mesh(g = {g})', ft.p2_mesh(g, te)) for g in (0.5, 0.4, 0.3, 0.2, 0.14, 0.1)]


def timed_median(fn, budget):
    """Run fn; repeat (21 runs if it took < 0.05 s, 7 if < 0.2 s, 3 if < `budget` s) and return
    (result, median s).  The many repeats steady the millisecond timings on a busy machine."""
    out, t = bc.timed(fn)
    n = 21 if t < 0.05 else 7 if t < 0.2 else 3 if t < budget else 1
    if n > 1:
        out, t = bc.median_time(fn, n)
    return out, t


def halved(params):
    """The same mesh recipe with every element size halved: the corner size hc, the sizes at the
    substrate surface (h_0) and the film top (h_tp), the caps hx_max and hy_max, and the growth rate
    (sizes grow by growth x distance from the key lines, so halving it halves the graded sizes too).
    y_far, the length over which the y cap itself grows, is not an element size and is unchanged."""
    q = dict(params)
    for k in ('hc', 'h_0', 'h_tp', 'hx_max', 'hy_max', 'growth'):
        q[k] = q[k] / 2
    return q


def run(params):
    """Setup and one solve, each timed as a median (timed_median); returns (cell, S, t_setup, t_solve)."""
    cell, t_setup = timed_median(lambda: ft.QuarterCell(**params, **GEOM), 1.0)
    S, t_solve = timed_median(lambda: cell.S(bc.EPS_FILM, bc.EPS_SUB, 'primal'), 5.0)
    return cell, S, t_setup, t_solve


def main():
    bc.banner('bench_fem_quadratic.py (d2): biquadratic FEM (fem_tensor.py, p = 2), upper bound on S')
    print(f"{'rung':38s} {'nodes':>9s} {'S':>12s} {'S/S_REF-1':>10s} {'setup s':>9s} {'solve s':>9s}  beats model")
    done = []
    for label, params in RUNGS:
        cell, S, t_setup, t_solve = run(params)
        print(f'{label:38s} {cell.n_nodes:9,d} {S:12.9f} {bc.rel_err(S):+10.2e} {t_setup:9.4f} {t_solve:9.4f}  '
              f"{'yes' if S <= bc.S_MODEL else 'no'}", flush=True)
        bc.record('FEM biquadratic (upper bound)', label, cell.n_nodes, t_setup, t_solve, S,
                  per_eval_s=t_solve, notes=f"p = 2 fem_tensor.py; {', '.join(f'{k}={v:.4g}' for k, v in params.items() if k != 'p')}",
                  quiet=True)
        done.append(dict(label=label, params=params, n=cell.n_nodes, S=S))
        del cell

    # convergence check of the cheapest rung that beats the model (see the module docstring)
    base = bc.fewest_nodes(done, lambda r: bc.as_accurate_as_model(r['S']))
    if base is None:
        print('No rung reached the model accuracy: no convergence check.')
        return
    q = halved(base['params'])
    cell, S, t_setup, t_solve = run(q)
    print(f"{bc.check_label(base['label'])[:38]:38s} {cell.n_nodes:9,d} {S:12.9f} {bc.rel_err(S):+10.2e} "
          f"{t_setup:9.4f} {t_solve:9.4f}", flush=True)
    bc.record_check('FEM biquadratic (upper bound)', base['label'], base['S'], cell.n_nodes, t_setup, t_solve, S,
                    extra_notes='p = 2 fem_tensor.py; ' + ', '.join(f'{k}={v:.4g}' for k, v in q.items() if k != 'p'))


if __name__ == '__main__':
    main()
