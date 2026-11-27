"""
reference_bounds.py -- the rigorous bracket  S_LOWER <= S <= S_UPPER  for the benchmark problem, from
biquadratic finite elements (fem_tensor.py) on a ladder of fine meshes, and the lines to paste into
bench_common.py.

WHAT IT DOES
  For each level (a growth value g of the mesh family fem_tensor.p2_mesh; finer meshes for smaller g):
    * upper bound: primal FEM, potential V (Dirichlet's principle: the energy of ANY potential that
      obeys the boundary conditions is at least the true energy);
    * lower bound: dual FEM, stream function psi of the displacement field (Thomson's principle).
  Every level gives a rigorous bracket, so the best one is the largest lower bound with the smallest
  upper bound over all levels.  S_REF is its midpoint: not guaranteed, but on every mesh tried the two
  bounds missed the true value by nearly equal and opposite amounts (along this ladder the midpoint
  moved by 1.2e-10 relative while the width shrank from 3.2e-8 to 1.4e-9).
  The bounds are for the problem exactly as posed in bench_common.py, with V = 0 at y = +-3a
  (an infinite cell has S lower by 1.0e-8 relative: measured with the boundary at 4a and 5a).

HOW TO RUN  (needs numpy and scipy; one thread, as for all the benchmarks)
    python reference_bounds.py                          # default levels: finest 4.7 million nodes
    python reference_bounds.py --levels 0.14,0.1,0.08   # quick check, about 30 s
    python reference_bounds.py --max-nodes 2e6          # skip the levels above 2 million nodes
    python reference_bounds.py --cluster                # larger levels for the BYU cluster
                                                        # (see reference_cluster.sh)
  Writes results/reference.txt and appends two rows per level to results/benchmark_results.csv.

EXPECTED OUTPUT  (Claude's sandbox: Intel Xeon 2.1 GHz shared with other jobs, 21 Nov 2026; run with
--levels 0.14,0.1,0.08,0.07,0.065 because the sandbox has only ~3 GB to spare; 83 s in all)
     g    hc/t_e      nodes         S_lower         S_upper     width        midpoint  setup s  solve s  peak GB
 0.140  6.70e-04    269,885   7.18810220083   7.18810242865  3.17e-08   7.18810231474      0.3      3.0     0.56
 0.100  2.44e-04    586,639   7.18810228779   7.18810234200  7.54e-09   7.18810231490      0.7      8.1     1.08
 0.080  1.25e-04    977,347   7.18810230477   7.18810232605  2.96e-09   7.18810231541      1.2     14.2     1.88
 0.070  8.37e-05  1,325,007   7.18810230901   7.18810232197  1.80e-09   7.18810231549      2.0     22.2     2.35
 0.065  6.70e-05  1,566,025   7.18810231070   7.18810232044  1.35e-09   7.18810231557      2.1     28.8     2.67
    Tightest bracket: 7.188102310698 <= S <= 7.188102320436  (width 1.35e-09 relative)
    Model (ide_model.py default settings) S_MODEL = 7.1889188: +0.01136% to +0.01136% from the true S
    Lines for bench_common.py (bracket rounded outward at the 11th decimal):
        S_LOWER = 7.18810231069
        S_UPPER = 7.18810232044
        S_REF = 7.18810231557
  (solve s = both bounds; peak GB = the process's peak so far, i.e. that of the largest level.)

RUN TIME AND MEMORY  (the finest level dominates; our desktop PC is assumed about 2x faster per core)
  Measured here up to 1.6 million nodes: about 9 s per bound per million nodes, growing as
  N^1.15; peak memory about 1.7-1.9 GB per million nodes, nearly all of it SuperLU's factors
  (88 to 104 nonzeros per node, slowly growing with N; 17-21 bytes per nonzero all told).
  Extrapolated (time as N^1.4 to be safe; nonzeros per node + 9 per e-fold of N):
    default levels (finest g = 0.04, 4.7M nodes): 9-11 GB; about 7 min here, 3-4 min on our desktop PC;
        expected width about 2e-10
    --cluster (finest g = 0.025, 13.6M nodes): about 34 GB; about 25 min at the sandbox's speed;
        expected width about 4e-11
  Memory, not time, limits the bracket: the width falls as about N^-1.65 to N^-1.84.
  On Windows, SuperLU reserves (commits) about 20 times the matrix size as virtual memory up front
  and touches only what it needs; if Windows cannot commit that much, SuperLU retries with smaller
  blocks, and growing them later costs some extra memory while it copies.  The peak memory printed
  here is the real (resident) peak: Linux ru_maxrss, Windows PeakWorkingSetSize.

THE MESHES  (fem_tensor.p2_mesh; from the convergence study of Nov 2026)
  fem2d.py's recipe (mesh lines graded from the electrode faces and material boundaries) with:
  corner size hc = (t_e/4096)(g/0.1)^3 (the corner error goes as hc^(4/3), the grading error about
  as g^4), 16 hc at the substrate surface (weaker singularity there), caps hx_max = 3.5g and
  hy_max = 3g um, film-top spacing 0.3g um, and a y cap that grows away from the electrode plane
  (y_far = 16 um).  With biquadratic elements the far field matters: with fem2d's uniform caps
  (0.5 and 4 um) the far field, not the corners, limits the accuracy; the growing y cap gives the
  same width with about 30% fewer nodes.  Corner elements are tiny (hc = 3.6e-6 um at g = 0.04),
  which is fine in double precision: one step of iterative refinement changes S by ~1e-15.

LINEAR SOLVER  (fem_tensor.solve_spd)
  SuperLU with minimum-degree ordering on the symmetric structure (MMD_AT_PLUS_A) and diagonal
  pivots, i.e. effectively a sparse Cholesky factorization.  Compared with scipy's default ordering
  (COLAMD), measured here: bilinear 0.16M nodes: factors 13.5M vs 23.2M nonzeros, 0.74 vs 1.20 s,
  peak 0.45 vs 0.67 GB; bilinear 0.60M nodes: 58M vs 102M, 3.3 vs 7.5 s, 1.7 vs 2.7 GB;
  biquadratic 0.22M nodes: 18.7M vs 42.1M, 1.1 vs 3.0 s, 0.6 vs 1.2 GB.
"""
import bench_common as bc       # first: pins numpy/scipy to one thread

import argparse
import math
import os
import time

import fem_tensor as ft

DEFAULT_LEVELS = [0.14, 0.1, 0.08, 0.065, 0.055, 0.047, 0.04]
CLUSTER_LEVELS = [0.14, 0.1, 0.07, 0.05, 0.04, 0.035, 0.03, 0.025]


def geometry():
    """The benchmark problem's geometry, from bench_common."""
    return dict(s=bc.S_GAP, w=bc.W_FINGER, te=bc.T_E, tp=bc.T_P, y_extent_in_a=bc.Y_EXTENT_IN_A)


def one_level(g):
    """Upper and lower bound for mesh level g.  Each bound gets its own freshly built mesh and
    matrices (about 1 s per million nodes), released before the factorization, so that only one
    factorization is ever in memory."""
    params = dict(ft.p2_mesh(g, bc.T_E), **geometry())
    out = dict(g=g, hc=params['hc'])
    for form in ('primal', 'dual'):
        t0 = time.perf_counter()
        cell = ft.QuarterCell(**params)
        t1 = time.perf_counter()
        out[form] = cell.S(bc.EPS_FILM, bc.EPS_SUB, form, release=True)
        t2 = time.perf_counter()
        out[form + '_setup'], out[form + '_solve'] = t1 - t0, t2 - t1
        out[form + '_lu_nnz'] = cell.last['lu_nnz']
        out['n_nodes'] = cell.n_nodes
        del cell
    out['peak_mb'] = ft.peak_memory_mb()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Rigorous bracket of S from biquadratic FEM bounds.')
    ap.add_argument('--levels', type=str, default=None,
                    help='comma-separated growth values g, coarse to fine (default %s)'
                         % ','.join(map(str, DEFAULT_LEVELS)))
    ap.add_argument('--cluster', action='store_true',
                    help='use the larger cluster levels %s' % ','.join(map(str, CLUSTER_LEVELS)))
    ap.add_argument('--max-nodes', type=float, default=None, help='skip levels with more nodes')
    args = ap.parse_args(argv)
    levels = ([float(v) for v in args.levels.split(',')] if args.levels
              else CLUSTER_LEVELS if args.cluster else DEFAULT_LEVELS)
    bc.banner('reference_bounds.py: rigorous bracket of S from biquadratic FEM (primal + dual)')
    bc.keep_awake()

    rows = []
    head = (f"{'g':>6} {'hc/t_e':>9} {'nodes':>10} {'S_lower':>15} {'S_upper':>15} {'width':>9} "
            f"{'midpoint':>15} {'setup s':>8} {'solve s':>8} {'peak GB':>8}")
    print(head, flush=True)
    for g in levels:
        n = ft.count_nodes(**dict(ft.p2_mesh(g, bc.T_E), **geometry()))
        if args.max_nodes and n > args.max_nodes:
            print(f'{g:6.3f}  skipped: {n:,} nodes > --max-nodes', flush=True)
            continue
        r = one_level(g)
        lo, up = r['dual'], r['primal']
        line = (f"{g:6.3f} {r['hc'] / bc.T_E:9.2e} {r['n_nodes']:10,d} {lo:15.11f} {up:15.11f} "
                f"{up / lo - 1:9.2e} {0.5 * (lo + up):15.11f} "
                f"{r['primal_setup'] + r['dual_setup']:8.1f} {r['primal_solve'] + r['dual_solve']:8.1f} "
                f"{(r['peak_mb'] or float('nan')) / 1024:8.2f}")
        print(line, flush=True)
        rows.append((r, line))
        label = f'p2_mesh(g = {g})'
        notes = f"width {up / lo - 1:.2e}; LU nonzeros {r['primal_lu_nnz']:,}"
        if r['peak_mb']:
            notes += f"; process peak memory {r['peak_mb']:.0f} MB"
        bc.record('FEM biquadratic upper bound (reference)', label, r['n_nodes'], r['primal_setup'],
                  r['primal_solve'], up, per_eval_s=r['primal_solve'], notes=notes, quiet=True)
        bc.record('FEM biquadratic lower bound (dual, reference)', label, r['n_nodes'], r['dual_setup'],
                  r['dual_solve'], lo, per_eval_s=r['dual_solve'], notes=notes, quiet=True)
    if not rows:
        return

    # best bracket over all levels (each one is rigorous), rounded outward at the 11th decimal
    lower = max(r['dual'] for r, _ in rows)
    upper = min(r['primal'] for r, _ in rows)
    S_lower = math.floor(lower * 1e11) / 1e11
    S_upper = math.ceil(upper * 1e11) / 1e11
    S_ref = 0.5 * (lower + upper)
    finest = max(rows, key=lambda t: t[0]['n_nodes'])[0]
    paste = [f'S_LOWER = {S_lower:.11f}',
             f'S_UPPER = {S_upper:.11f}',
             f'S_REF = {S_ref:.11f}']
    summary = [
        '',
        f'Tightest bracket: {lower:.12f} <= S <= {upper:.12f}  (width {upper / lower - 1:.2e} relative)',
        f"Finest level: g = {finest['g']}, {finest['n_nodes']:,} nodes; process peak memory "
        f"{(finest['peak_mb'] or float('nan')) / 1024:.2f} GB",
        f'Model (ide_model.py default settings) S_MODEL = {bc.S_MODEL:.7f}: '
        f'{100 * (bc.S_MODEL / upper - 1):+.5f}% to {100 * (bc.S_MODEL / lower - 1):+.5f}% from the true S',
        '(Bounds for the problem as posed: V = 0 at y = +-3a.)',
        '',
        'Lines for bench_common.py (bracket rounded outward at the 11th decimal):',
    ] + ['    ' + p for p in paste]
    print('\n'.join(summary), flush=True)

    m = bc.machine_info()
    with open(os.path.join(bc.RESULTS_DIR, 'reference.txt'), 'w') as f:
        f.write('reference_bounds.py  ' + time.strftime('%Y-%m-%d %H:%M') + '\n')
        f.write(f"{m['cpu']} | {m['os']} | Python {m['python']}, numpy {m['numpy']}, scipy {m['scipy']}\n\n")
        f.write(head + '\n' + '\n'.join(line for _, line in rows) + '\n')
        f.write('\n'.join(summary) + '\n')
    print(f"\nWrote {os.path.join(bc.RESULTS_DIR, 'reference.txt')}")


if __name__ == '__main__':
    main()
