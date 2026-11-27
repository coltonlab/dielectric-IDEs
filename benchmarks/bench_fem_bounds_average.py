"""
bench_fem_bounds_average.py (d3) -- bilinear finite elements (fem_tensor.py, p = 1): upper AND lower
bound on the same mesh, and their average.

On fem2d.py-style meshes (h_c, growth) = (t_e, 0.6) ... (t_e/128, 0.02), with fem2d's caps
hx_max = 0.5 um and hy_max = 4 um (the last five are the meshes of the paper's Supplemental Section S10):
  * upper bound: primal FEM, potential V (the same S as fem2d.py, to rounding: verify_fem_tensor.py)
  * lower bound: dual FEM, stream function psi of the displacement field (Thomson's principle)
  * average of the two.
The bracket lower <= S <= upper is rigorous.  The AVERAGE IS NOT GUARANTEED.  It is far more
accurate than either bound here, because the two errors are nearly equal and opposite, but nothing
forces this: it crosses the true S near 50,000 nodes and then settles about 6e-7 BELOW it (near
7.188097-7.188098), because fem2d's fixed caps leave a small error that refining h_c and the growth
does not remove.  So the stability of the average is no evidence of its accuracy; only the bracket
is a guarantee.

Recorded rows: 'FEM bound average (bilinear)' (solve time = both solves) and
'FEM lower bound (bilinear, dual)'.  The upper bound is fem2d's value and is not recorded again here.
Times are medians of 3 runs when a solve takes less than 5 s; setup = mesh + per-region matrices
(shared by both bounds).

    python bench_fem_bounds_average.py      (about 40 s here)

Expected output (Claude's sandbox, Intel Xeon 2.1 GHz shared with other jobs, 21 Nov 2026).  The
error column is relative to bench_common.S_REF: shown here for S_REF = 7.18810231557 from
reference_bounds.py (with the older provisional S_REF = 7.1880977 it reads 6.4e-7 higher).
Times vary by about +-20% between runs.
    mesh                           nodes        lower        upper     width      average avg/S_REF-1  setup s  lower s  upper s
    h_c = t_e/1, growth 0.6       10,707  7.173090018  7.204662060  4.40e-03  7.188876039   +1.08e-04    0.009    0.023    0.021
    h_c = t_e/2, growth 0.5       11,832  7.179930597  7.196496902  2.31e-03  7.188213750   +1.55e-05    0.011    0.024    0.023
    h_c = t_e/4, growth 0.3       14,194  7.183264030  7.193016494  1.36e-03  7.188140262   +5.32e-06    0.007    0.030    0.028
    h_c = t_e/8, growth 0.2       18,725  7.184965278  7.191280175  8.79e-04  7.188122727   +2.84e-06    0.013    0.046    0.037
    h_c = t_e/16, growth 0.12     30,375  7.186421722  7.189787902  4.68e-04  7.188104812   +3.47e-07    0.013    0.069    0.067
    h_c = t_e/32, growth 0.06     75,402  7.187538127  7.188660742  1.56e-04  7.188099434   -4.01e-07    0.032    0.235    0.254
    h_c = t_e/64, growth 0.04    162,447  7.187828074  7.188365432  7.48e-05  7.188096753   -7.74e-07    0.077    0.765    0.671
    h_c = t_e/128, growth 0.02   598,502  7.188025395  7.188169963  2.01e-05  7.188097679   -6.45e-07    0.335    3.661    3.824
The first rung's average is about as accurate as the model (+1.14e-4), but its bracket is 4e-3 wide:
its accuracy is luck, not something one could know without a reference.

CONVERGENCE CHECK.  After the ladder, the rung with the fewest nodes whose average is at least as
accurate as the model (|error| <= the model's error, judged with S_REF, as in plot_benchmarks.py) is
solved once more -- both bounds -- with every element size halved (h_c, the growth, and the caps
hx_max = 0.5 um, hy_max = 4 um), timed in the same way and logged as 'check: <rung label>' under
'FEM bound average (bilinear)' (bench_common.record_check; S = the average on the check mesh).
Claude's sandbox, 21 Nov 2026 (shared): check of h_c = t_e/1, growth 0.6 -> h_c = t_e/2, growth 0.3,
caps 0.25/2 um: 42,994 nodes, average 7.1881923338 (bracket width 8.4e-4); estimate +9.51e-05 (true
+1.08e-04); setup + both solves 0.34 s against 0.061 s for the rung (x6.5 for first S + check).
"""
import bench_common as bc       # first: pins numpy/scipy to one thread

import fem_tensor as ft

GEOM = dict(s=bc.S_GAP, w=bc.W_FINGER, te=bc.T_E, tp=bc.T_P, y_extent_in_a=bc.Y_EXTENT_IN_A)
LADDER = [(1, 0.6), (2, 0.5), (4, 0.3),                                     # (t_e / h_c, growth)
          (8, 0.20), (16, 0.12), (32, 0.06), (64, 0.04), (128, 0.02)]
HX_MAX, HY_MAX = 0.5, 4.0          # fem2d.py's caps (and QuarterCell's defaults), used by every rung above


def bounds(div, growth, hx_max=HX_MAX, hy_max=HY_MAX):
    """Setup (timed once) and both solves (each timed once, then the median of 3 if under 5 s);
    returns (cell, lower, upper, t_setup, t_lower, t_upper)."""
    cell, t_setup = bc.timed(lambda: ft.QuarterCell(p=1, hc=bc.T_E / div, growth=growth, hx_max=hx_max,
                                                    hy_max=hy_max, **GEOM))
    times = {}
    S = {}
    for form in ('dual', 'primal'):
        S[form], times[form] = bc.timed(cell.S, bc.EPS_FILM, bc.EPS_SUB, form)
        if times[form] < 5.0:
            S[form], times[form] = bc.median_time(cell.S, 3, bc.EPS_FILM, bc.EPS_SUB, form)
    return cell, S['dual'], S['primal'], t_setup, times['dual'], times['primal']


def main():
    bc.banner('bench_fem_bounds_average.py (d3): bilinear FEM, upper and lower bound and their average')
    print(f"{'mesh':26s} {'nodes':>9s} {'lower':>12s} {'upper':>12s} {'width':>9s} {'average':>12s} "
          f"{'avg/S_REF-1':>11s} {'setup s':>8s} {'lower s':>8s} {'upper s':>8s}")
    done = []
    for div, growth in LADDER:
        label = f'h_c = t_e/{div}, growth {growth}'
        cell, lo, up, t_setup, t_lo, t_up = bounds(div, growth)
        avg = 0.5 * (lo + up)
        print(f'{label:26s} {cell.n_nodes:9,d} {lo:12.9f} {up:12.9f} {up / lo - 1:9.2e} {avg:12.9f} '
              f"{bc.rel_err(avg):+11.2e} {t_setup:8.3f} {t_lo:8.3f} {t_up:8.3f}", flush=True)
        bc.record('FEM bound average (bilinear)', label, cell.n_nodes, t_setup,
                  t_lo + t_up, avg, per_eval_s=t_lo + t_up,
                  notes=f'average of lower {lo:.9f} and upper {up:.9f} (not a guaranteed bound)', quiet=True)
        bc.record('FEM lower bound (bilinear, dual)', label, cell.n_nodes, t_setup, t_lo, lo,
                  per_eval_s=t_lo, notes='rigorous lower bound (Thomson)', quiet=True)
        done.append(dict(label=label, div=div, growth=growth, n=cell.n_nodes, S=avg))
        del cell

    # convergence check of the cheapest rung as accurate as the model (see the module docstring); the
    # average is not a bound, so "as accurate" compares the size of the error, as plot_benchmarks.py does
    err_model = bc.rel_err(bc.S_MODEL)
    base = bc.fewest_nodes(done, lambda r: abs(bc.rel_err(r['S'])) <= err_model)
    if base is None:
        print('No rung reached the model accuracy: no convergence check.')
        return
    div2, growth2, hx, hy = 2 * base['div'], base['growth'] / 2, HX_MAX / 2, HY_MAX / 2
    cell, lo, up, t_setup, t_lo, t_up = bounds(div2, growth2, hx, hy)
    avg = 0.5 * (lo + up)
    print(f"{bc.check_label(base['label'])[:26]:26s} {cell.n_nodes:9,d} {lo:12.9f} {up:12.9f} {up / lo - 1:9.2e} "
          f"{avg:12.9f} {bc.rel_err(avg):+11.2e} {t_setup:8.3f} {t_lo:8.3f} {t_up:8.3f}", flush=True)
    bc.record_check('FEM bound average (bilinear)', base['label'], base['S'], cell.n_nodes, t_setup, t_lo + t_up,
                    avg, extra_notes=f'h_c = t_e/{div2:g}, growth {growth2:g}, hx_max {hx:g} um, hy_max {hy:g} um; '
                                     f'average of lower {lo:.9f} and upper {up:.9f}; bracket width {up / lo - 1:.2e}')


if __name__ == '__main__':
    main()
