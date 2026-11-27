# Benchmarks: our model versus other ways of computing the capacitance

This folder holds the computing-time comparison of Supplemental Section S3 of the paper (Table S1 and
Fig. S3) and independent checks of the repository's finite-element code `fem2d.py`. Written with Claude
(Anthropic), Nov 2026. Everything runs on one processor core.

## What this folder is for

1. **Speed at equal accuracy.** Compare our model (`../ide_model.py`) with standard numerical methods on
   one test problem: how long each method takes to reach our model's accuracy, for one capacitance and
   for the extraction of all the paper's dielectric functions.
2. **Independent checks of `fem2d.py`.** New finite-element and finite-difference code, written
   separately, reproduces fem2d.py. Rigorous lower bounds bracket the true value. A completely different
   method (relaxation on uniform grids) converges to the same answer.

## Where the paper's numbers come from

| In the paper | Made by | From |
|---|---|---|
| Table S1 (time to extract the dielectric constants at the 78 frequencies) | `core_extraction_timing.py`, then `table_s1.py` | `results/core_extraction.txt`; the relaxation rows also use `results/benchmark_results.csv` |
| Fig. S3 (error against computing time) | the `bench_*.py` scripts (run by `run_daytime.py` and `run_overnight.py`), then `fig_s3.py` | `results/benchmark_results.csv` |
| Reference value (fixed to 2 parts in 10^10) and our model's error (+0.0114%) | `reference_bounds.py` | `results/reference.txt` |
| Times of a single capacitance in Sec. 4.1 and Section S3 (1.4 ms, 6.3 ms, 0.3 s; about 5× and 220×, or 40× and 2,300× with the mesh reused; about 4 min by relaxation on a uniform grid) | the `bench_*.py` scripts; `plot_benchmarks.py` summarizes them | `results/benchmark_results.csv`, `results/summary.txt` |
| Numbers of evaluations per root solve and per film, times per evaluation (Section S3) | `core_extraction_timing.py` | `results/core_extraction.txt` |

All the results in `results/` were measured on our desktop PC (Intel Core Ultra 7 265, Windows 11,
Python 3.13, one core) between 21 and 24 Nov 2026. `table_s1.py` and `fig_s3.py` reproduce Table S1 and
Fig. S3 from them in seconds, without rerunning the benchmarks.

## The test problem (identical for every method)

- **Electrodes:** the paper's IDE: gap s = 50.27 µm, finger width w = 29.78 µm, electrode thickness
  t_e = 115 nm.
- **Film and substrate:** a MAPbI3-like film (ε = 47, thickness t_p = 437.8 nm measured from the
  substrate surface) on quartz (ε = 4.53).
- **Grid methods:** they solve one quarter of a period, with V = 0 at y = ±3a. Moving that
  boundary to 4a or 5a changes S by 1e-8.
- **Output:** the dimensionless capacitance S, normalized as in fem2d.py and ide_model.py
  (C = prefactor × S).
- **Settings:** all numbers are in `bench_common.py`.

**Reference value.** `reference_bounds.py` computes two bounds on the same mesh with biquadratic
elements. The upper bound comes from Dirichlet's principle (potential V). The lower bound comes from
Thomson's principle (stream function ψ). On a 4.7-million-node mesh (`results/reference.txt`):

    7.18810231472  <=  S_true  <=  7.18810231618      (width 2.0e-10)

Against this bracket:

- our model (default settings) is **+0.01136 %** high;
- fem2d.py on the paper's finest mesh is **+0.0037 %** high.

## Files

| Script | Item | Method | Equations | Solver |
|---|---|---|---|---|
| `bench_model.py` | (a) | our model | Fourier series + optimized in-plane potential | ~40×40 linear solve |
| `bench_sor_graded.py` | (b1) | relaxation | 5-point (linear-triangle) equations, graded grids | red-black SOR (Numba) |
| `bench_direct_graded.py` | (b2) | direct | same equations and grids as b1 | SuperLU |
| `bench_fem2d.py` | (b3) | fem2d.py | bilinear FEM, the meshes of Supplemental Sections S4 and S10 | SuperLU |
| `bench_sor_squares.py` | (c1) | bare-bones relaxation | 5-point, uniform squares* | red-black SOR |
| `bench_sor_rectangles.py` | (c2) | bare-bones relaxation | 5-point, uniform 2:1 rectangles* | red-black SOR |
| `bench_fem2d_cheapest.py` | (d1) | fem2d.py, tuned mesh | bilinear FEM, cheapest mesh matching our model | SuperLU |
| `bench_fem_quadratic.py` | (d2) | higher-order FEM | biquadratic FEM (upper bound) | SuperLU |
| `bench_fem_bounds_average.py` | (d3) | bound average | mean of bilinear upper and lower bounds | SuperLU |
| `reference_bounds.py` | | reference | biquadratic upper and lower bounds | SuperLU |

\* Each interval between interfaces is divided into equal parts. Cells are therefore uniform to
within a few per cent, and every interface lies exactly on a grid line. Rounding the geometry to a
strictly uniform grid would be far too costly: moving the film top by 7 nm changes S by 0.26 %.

Timing of the paper's analysis:

- `core_extraction_timing.py`: times our model's extraction of all the paper's data (the two blanks and
  the 13 film measurements, 78 frequencies, with and without the complex fit below 2 MHz), counts the
  evaluations of the capacitance, times one capacitance with each direct method, and estimates the
  extraction time of each method. Writes `results/core_extraction.txt`.
- `table_s1.py`: Table S1 from `results/core_extraction.txt` and `results/benchmark_results.csv`.
- `whole_analysis_timing.py`: a broader count (not used in the paper): the model's work in
  `reproduce_paper.py`, `uncertainty.py --paper` and `kramers_kronig.py --paper` together, run on a
  temporary copy of the repository. Writes `results/whole_analysis.txt`.

Supporting files:

- **Libraries:**
  - `bench_common.py`: the problem, the reference, timing and the results log. It finds `ide_model.py`
    and `fem2d.py` in the parent folder (the repository); the environment variable `DIELECTRIC_IDES`
    can point to another copy.
  - `fem_tensor.py`: bilinear or biquadratic FEM, upper or lower bound; independent of fem2d.py.
  - `fivepoint.py`: the 5-point equations on any rectangular grid, the direct solver and the
    compiled SOR.
- **Checks:** `verify_fem_tensor.py` and `verify_fivepoint.py`.
- **Background experiments:** `experiments_fivepoint.py` reruns the experiments behind the choices
  for ω, the grid families and the aspect ratio.
- **Runners and plots:**
  - `run_daytime.py`: everything quick;
  - `run_overnight.py`: the long SOR runs, within an 8-hour budget;
  - `plot_benchmarks.py`: the comparison figure `results/benchmark_comparison.png` (with all methods'
    convergence checks) and the summary table `results/summary.txt`;
  - `fig_s3.py`: Supplemental Fig. S3 (`results/Fig_S3_benchmarks.png` and `.pdf`).
- **Cluster:** `reference_cluster.sh` (optional; see the notes at the end).

## How to run

The commands below use `py`, the Python launcher on Windows; elsewhere use `python3`.

1. Install the packages once, from the repository folder (numba is needed by the five-point scripts,
   pandas only by `fig_s3.py`):

       py -m pip install -r benchmarks/requirements.txt

   Numba 0.68 works with Python 3.13 and numpy below 2.6.
2. Run from the `benchmarks` folder:

       cd benchmarks
       py table_s1.py                 # Table S1 from the results in results/ (instant)
       py fig_s3.py                   # Fig. S3 from the results in results/ (a few seconds)
       py core_extraction_timing.py   # times the extraction; rewrites results/core_extraction.txt (under a minute)
       py run_daytime.py              # all quick benchmarks, checks and plots (30-50 min on our PC)
       py run_overnight.py            # the long SOR runs (up to 8 h; 7 h on our PC)
       py plot_benchmarks.py          # results/benchmark_comparison.png and results/summary.txt

   `run_daytime.py --skip-reference` leaves out `reference_bounds.py` (about 4 min and 8-10 GB of
   memory); `run_daytime.py --only bench_model.py bench_fem2d.py` runs only the scripts named. Each
   benchmark script also runs on its own, e.g. `py bench_model.py`; every script describes its options
   at the top and prints them with `--help` where it has any.
3. For reliable times, run on a plugged-in computer with nothing else heavy running (on Windows, set
   *Settings → System → Power → Power mode: Best performance*). The scripts keep Windows from sleeping
   while they run and opt out of its power throttling. `run_daytime.py` times a fixed calibration job
   at the start and at the end, to catch slowdowns.
4. New results are **appended** to `results/benchmark_results.csv` (one row per run) and
   `results/sor_trajectories.csv` (S against time during each SOR run); the logs, `summary.txt` and the
   figures are rewritten. `fig_s3.py` and `plot_benchmarks.py` use the newest row of each rung, so new
   runs replace ours in the plots. Copy `results/` first if you want to keep our runs.

## How the comparison is kept fair

- **Same problem, same boundary, one core.** Every library is pinned to one thread; SuperLU and the
  SOR loops are single-threaded anyway.
- **What is timed.**
  - Times are medians of repeats.
  - Imports and Numba compilation are not timed.
  - Every script reports two times:
    - **first S**: setup plus one solve;
    - **each further S**: the setup reused for new dielectric constants. This is what matters when
      a whole spectrum is analyzed.
- **Accuracy criterion that needs no reference.**
  - Every method except the bound average (d3) gives an upper bound on S. So "S ≤ S_model" means
    "at least as accurate as the model".
  - The SOR runs stop as soon as that happens. Runs whose converged S stays above it are run to
    convergence instead.
- **Tuning is not timed.** This is generous to the numerical methods:
  - the search for fem2d's cheapest mesh (d1);
  - the choice of the coarse biquadratic meshes (d2);
  - SOR's relaxation factor ω. On graded grids, ω comes from the exact spectral radius (scipy
    `eigsh`), computed before the timed run; on uniform grids, from a validated formula.
- **Knowing the error costs one more solve.** The rungs are judged against the reference value,
  which a user would not have. The least a user can do to learn a mesh's error is to solve once
  more on a finer mesh; this is timed separately as a convergence check (next section).

## Convergence checks

- **What.** After its ladder, each script takes the rung with the fewest nodes that is at least as
  accurate as the model and solves the same problem once more, by the same method, on a mesh or
  grid with **every element size halved**: the corner size h_c, the sizes at the substrate surface
  and the film top, the caps hx_max and hy_max, and the growth rate (the sizes grow by growth ×
  distance, so halving the growth halves the graded sizes too). y_far (d2) is a length, not a size,
  and is unchanged. For the 5-point graded grids this is refine × 2.
- **Timing.** Setup + one solve, with the same timing rules as that script's rungs. For SOR (b1),
  the check runs from V = 0 to convergence (remaining decrease below 1e-7 of S), because a user
  checking a grid does not know when to stop; ω again comes from `eigsh`, not timed.
- **Logged** as its own row: same method, label `check: <label of the rung>`
  (`bench_common.record_check`). The notes give the check's estimate of the rung's error,
  S_rung/S_check − 1 (what a user would see), and the true error.
- **Summary** (`plot_benchmarks.py`, `results/summary.txt`):
  - the main table gains `first+chk` (the rung's first S plus the check) and its ratio to the
    model's first S plus the model's own check;
  - a second table lists each check: rung and check nodes, estimate against true error, the cost
    ratio (first S + check)/(first S), and first S + check divided by the model's first S
    **without** a check;
  - the figure has a third panel with the first S of each method's checked rung (filled marker)
    and first S + check (open marker).
- **Our model's check** (`bench_model.py`): the default settings with the in-plane node spacing
  halved. It sees only the model's discretization error (about 1e-5), not its modelling error
  (about 1.1e-4), which is known only from comparisons such as these. So the fair comparison
  depends on the geometry:
  - inside the range of geometries where the paper establishes the model's error, the model needs
    no check;
  - outside it, neither the model nor its own check can tell the error.
- **What the checks show** (`results/summary.txt`). The biquadratic check estimates the rung's error
  almost exactly (its mesh family converges fast). The bilinear FEM and 5-point checks underestimate it
  by about 20-30 %, the bound average by about 12 %. A check multiplies the cost of a first S by about
  5 to 13.
- **No check for the uniform grids (c1, c2).** Their next grid would have about 93 million nodes.
  Their coarser rungs (t_e/1 … t_e/4) already show how S converges, a Richardson-type estimate.
- **Still a lower bound on the real cost.** The rung being checked was itself picked with the
  reference value. A user without it would land on some other mesh, usually a finer one, and
  might need more than one refinement to be sure.
- **Flags.** `bench_direct_graded.py` and `bench_sor_graded.py` take `--no-check`.
  `run_overnight.py` uses it for single b1 rungs; b1's check is part of `run_daytime.py`.

## Checks of fem2d.py

1. **Independent re-implementation.** `fem_tensor.py` reproduces fem2d.py's S to 1.5e-11 on the same
   meshes, and its potentials to 3e-11.
   - Its element matrices come from Gauss quadrature of the shape functions, not from fem2d's
     tables.
   - It has its own mesh generator.
   - The remaining 1e-11 is rounding in fem2d's evaluation of VᵀKV.
2. **Rigorous lower bounds.**
   - The dual (Thomson) formulation on fem2d's own meshes gives lower bounds.
   - fem2d's S always lies above them, and the gap closes as 1/N (4.7e-4 at 30k nodes, 2.0e-5 at
     600k nodes).
   - With biquadratic elements the bracket closes to 2.0e-10 (the reference value above).
3. **Exact solutions.**
   - A two-layer parallel plate is exact to 1e-16 (both formulations, both element orders).
   - Zero-thickness electrodes agree with the conformal-mapping result to 1.2e-6.
4. **A completely different method.** Relaxation of the 5-point equations on uniform grids uses no
   finite elements and no grading. Its converged S approaches the same value: +0.055 %, +0.022 %,
   +0.014 % and +0.0093 % at h = t_e, t_e/2, t_e/3 and t_e/4, with the error falling roughly as
   h^1.4.
5. **An earlier relaxation code reproduced, and SOR checked.**
   - The direct solve of the 5-point equations reproduces the results of relax_ide.py, an earlier
     relaxation code of ours that is not part of this repository (its values are quoted in
     `verify_fivepoint.py` and in the docstrings).
   - SOR converges to the direct solution to 1e-13.
   - S never increases during SOR, as the theory requires.

## The results in `results/`

Measured on our desktop PC (Intel Core Ultra 7 265, one core):

- `reference.txt`: the reference bracket (21 Nov 2026).
- `benchmark_results.csv` and `sor_trajectories.csv`: every benchmark run, 21-23 Nov 2026. The paper
  uses the newest run of each method: 22 Nov for our model and the direct methods, 23 Nov for
  relaxation (a rerun; the uniform-grid relaxation runs of 21-22 Nov took about three times as long per
  node update).
- `daytime_log.txt`, `overnight_log.txt`: what the runners printed.
- `core_extraction.txt` (24 Nov 2026): the timing behind Table S1. Its label "Appendix C meshes" is
  the earlier name of the meshes of Supplemental Section S4. Evaluation counts can differ slightly on
  another computer, because the root finders' iterations depend on rounding: in a Linux cloud machine
  the extraction took 9,924 real and 10,914 complex evaluations instead of 9,935 and 10,893.
- `whole_analysis.txt`, `whole_analysis_log.txt` (22 Nov 2026): the broader count of
  `whole_analysis_timing.py`.
- `summary.txt`, `benchmark_comparison.png` and `.pdf`: made by `plot_benchmarks.py` from
  `benchmark_results.csv` (newest row of each rung).
- `Fig_S3_benchmarks.png` and `.pdf`: Supplemental Fig. S3, made by `fig_s3.py`.

The docstrings of the scripts quote "expected" outputs from Claude's sandbox, the cloud machine in which
the scripts were developed (an Intel Xeon at 2.1-2.3 GHz, shared with other jobs). Times there are
about 1.3 to 2 times longer than on our PC.

## Notes and caveats

- **Our model was sped up on 21 Nov 2026.** `ide_model.py` now evaluates its Fourier sums exactly:
  the modes beyond the first ~800 see the film as infinitely thick, and the sum over all of them has a
  closed form, computed once per geometry. Before, the sums were cut off at 20,000 terms. For the test
  problem S rises by 1.9e-8 (relative); setup and each S are about 20 times faster.
  `bench_common.S_MODEL` is the new default. `bench_model.py` also has a `legacy` rung with the sums
  cut off at 20,000 terms, run with the new code; it is logged but left out of the summary and figures.
- **The bound average (d3) is not a bound.** Only the bracket is guaranteed. With bilinear elements
  the average settles about 6.5e-7 below the true S.
- **Some choices used knowledge of the answer.**
  - The coarse biquadratic meshes of d2 and the tuned fem2d mesh of d1 were chosen by comparing
    with the model's S.
  - A user without that knowledge would have to run a short mesh-convergence study. Its least
    part, one solve with every element size halved, is timed (first S + check).
  - The model's own error is likewise known only from comparisons like these.
- **The tuned mesh of d1 depends on timings.** The scan's final pick, the fastest of the five
  smallest qualifying meshes, can differ between runs (hmax 1/16 µm on 21 Nov, 2/16 µm on 22 Nov).
  Its check follows whichever mesh is logged.
- **Memory.**
  - `reference_bounds.py` (default levels, finest 4.7M nodes) needs about 8-10 GB.
  - The c1 rung h = t_e/16 needs 9.3 GB and runs overnight only if it fits in the time budget.
  - On Windows, SuperLU reserves virtual memory generously. If a large solve fails with a memory
    error, use `--max-nodes` (reference_bounds.py) or skip that rung.
- **The cluster is not needed.** The bracket from a PC (width 2.0e-10) is already far narrower than
  any error compared here. `reference_cluster.sh` can repeat the reference on a Slurm cluster with
  finer levels. Its module names and virtual-environment path are placeholders, to be filled in from
  `module avail python`.
