"""
bench_sor_rectangles.py  (c2) -- red-black SOR on UNIFORM RECTANGULAR grids, hx = 2 hy.

Method 'SOR, uniform rectangles'.  Same as bench_sor_squares.py (c1), but with flat cells:
hy = t_e/m and hx = 2 t_e/m.  Each rung runs from V = 0 (electrode at V = 1) to S <= S_MODEL, or to
convergence if the grid's own S is above S_MODEL.

    python bench_sor_rectangles.py                        (default rungs; about 35 min on our desktop PC)
    python bench_sor_rectangles.py --rungs 6 --budget-hours 1
    python bench_sor_rectangles.py --ratio 0.5 --rungs 2  (other aspect ratios, for tests)

WHY hx/hy = 2: the most efficient non-square ratio of {1/4, 1/2, 2, 4} -- and squares are better
still.  Converged SOR at equal node count (5.8 million nodes, the equivalent of squares h = t_e/2):
    hx/hy     S             error      sweeps to 1e-5   1 - rho_J (relative to squares)
    1         7.18969422    +0.0222%   2,550            1
    2         7.18996595    +0.0260%   2,900            0.80
    1/2       7.19010298    +0.0279%   2,800            0.80
    4         7.19070799    +0.0363%   3,825            0.47
    1/4       7.19074762    +0.0369%   3,625            0.47
(direct solves at 1.46 million nodes rank them the same way).  Flat cells (hx > hy) are more
accurate than tall ones because the thin film and the electrode need resolution in y; but at equal
cell area squares resolve the singular fields at the electrode corners best, so they win.  For
SOR, theory gives 1 - rho_J proportional to 2r/(1 + r^2) at equal cell area (r = hx/hy): squares
converge fastest, 2:1 needs sqrt(1.25) = 1.12x the sweeps, 4:1 1.46x -- as measured above.

RESULTS (Claude's sandbox, 21 Nov 2026): S_grid is the converged S of each grid.
  m   hy (nm)  hx (nm)   nodes          S_grid        error      sweeps          reaches S_MODEL?
  1   115.0    230.0        730,975    7.1949185     +0.0949%     1,375 (conv)  no     measured
  2    57.5    115.0      2,915,895    7.19098806    +0.0402%     2,700 (conv)  no     measured
  3    38.3     76.7      6,553,190    7.18985375    +0.0244%     4,025 (conv)  no     measured
  4    28.75    57.5     11,644,779    7.1892915     +0.0166%    ~5,400 (conv)  no     S from a windowed grid
  5    23.0     46.0     18,189,093    7.1889828     +0.0123%    ~6,800 (conv)  no     (9e-6 above the target)
  6    19.2     38.3     26,188,745    7.1887911     +0.0096%    ~6,100         YES    first rung
  7    16.4     32.9     35,639,903    7.1886621     +0.0079%    ~6,850         yes
  8    14.4     28.75    46,544,309    7.1885749     +0.0066%    ~7,750         yes
 10    11.5     23.0     72,716,347    7.18845       +0.0050%    ~9,450         yes    extrapolated
 12     9.6     19.2    104,698,591    7.18838       +0.0039%   ~11,300         yes    extrapolated
 16     7.2     14.4    186,174,450    7.18829       +0.0027%   ~15,000         yes    extrapolated
The windowed-grid S values (rungs 4-8) are exact solutions on grids identical to these for |y| < 3 um
and graded beyond, corrected for the measured effect of the grading (2.9e-5, known to ~1e-6 from
rungs 2, 3 and the squares).  First rung reaching the target: 26.2 million nodes and ~6,100 sweeps
(1.6e11 node-updates), against 23.3 million and 5,025 (1.17e11) for squares: squares are about
1.4x faster.  The experiments behind these numbers: experiments_fivepoint.py (rectangles, uniform).
"""
import bench_common as bc      # first: one thread for every numerical library

import argparse

import fivepoint as fp
from bench_sor_squares import run_ladder        # the same ladder runner as c1

METHOD = 'SOR, uniform rectangles'
RATIO = 2.0                                     # hx / hy
RUNGS = (1, 2, 3, 4, 5, 6, 7, 8)                # run_overnight.py adds 10, 12, 16 if they fit
EXPECTED_SWEEPS = {1: 1375, 2: 2700, 3: 4025, 4: 5400, 5: 6800, 6: 6100, 7: 6850, 8: 7750,
                   10: 9450, 12: 11300, 16: 15000}


def make_grid_fn(ratio):
    return lambda m: fp.uniform_grid(ratio * bc.T_E / m, bc.T_E / m)


def label_fn(ratio):
    return lambda m: f'hy = t_e/{m:g} = {1000 * bc.T_E / m:.2f} nm, hx = {ratio:g} hy'


grid, rung_label = make_grid_fn(RATIO), label_fn(RATIO)


def n_nodes(m):
    xs, ys = grid(m)
    return xs.size * ys.size


def expected_sweeps(m):
    return EXPECTED_SWEEPS.get(m, round(950 * m))


def main():
    p = argparse.ArgumentParser(description='c2: red-black SOR on uniform rectangular grids, hy = t_e/m, hx = ratio*hy')
    p.add_argument('--rungs', type=float, nargs='+', default=RUNGS, help='values of m (hy = t_e/m)')
    p.add_argument('--ratio', type=float, default=RATIO, help='hx/hy (default 2)')
    p.add_argument('--budget-hours', type=float, default=None, help='total time budget (default none)')
    p.add_argument('--ns-guess', type=float, default=None, help='ns per node-update for the first estimate (run_overnight.py)')
    a = p.parse_args()
    bc.banner(f'c2: bench_sor_rectangles.py -- red-black SOR on uniform rectangles hx = {a.ratio:g} hy, '
              f'hy = t_e/m, from V = 0 to S <= S_MODEL')
    g, lab = make_grid_fn(a.ratio), label_fn(a.ratio)
    run_ladder(METHOD, a.rungs, g, lab, lambda m: g(m)[0].size * g(m)[1].size,
               expected_sweeps if a.ratio == RATIO else (lambda m: round(1300 * m)), a.budget_hours, a.ns_guess)


if __name__ == '__main__':
    main()
