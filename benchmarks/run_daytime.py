"""
run_daytime.py -- run all the quick benchmarks one after another, each in its own Python process,
then make the comparison figure.  About 30-50 min on our desktop PC (50 min on 21 Nov 2026); the long SOR runs
are in run_overnight.py.  The benchmark scripts b1, b2, b3, d1, d2 and d3 (and bench_model.py) end with
a convergence check of their cheapest rung at the model's accuracy (bench_common.record_check), which
adds about 1 min in all (most of it b1's SOR on refine 6).

    python run_daytime.py                       (everything below, in this order)
    python run_daytime.py --skip-reference      (leave out reference_bounds.py: about 4 min, ~10 GB RAM)
    python run_daytime.py --only bench_model.py bench_fem2d.py

  verify_fem_tensor.py        checks of the new FEM code (patch test, fem2d.py reproduced, exact t_e = 0 case)
  verify_fivepoint.py         checks of the 5-point code (relax_ide.py reproduced, SOR = direct solve)
  bench_model.py              (a)  the paper's model
  bench_fem2d.py              (b3) fem2d.py, the paper's FEM
  bench_fem2d_cheapest.py     (d1) fem2d.py with the cheapest mesh that matches the model
  bench_fem_quadratic.py      (d2) biquadratic FEM
  bench_fem_bounds_average.py (d3) average of bilinear upper and lower bounds
  bench_direct_graded.py      (b2) 5-point equations on graded grids, direct solve
  bench_sor_graded.py         (b1) the same equations, SOR
  bench_sor_squares.py        (c1) SOR on uniform squares, rungs h = t_e/1 ... t_e/4 (about 7 min)
  bench_sor_rectangles.py     (c2) SOR on uniform 2:1 rectangles, rungs 1-6 (about 13 min)
  reference_bounds.py         the rigorous bracket of the true S, finer than the one in bench_common.py
  plot_benchmarks.py          figure and summary table

Everything printed goes to the screen and to results/daytime_log.txt.  The machine speed is measured
(bench_common.calibrate) at the start and at the end; a big difference means the PC was throttled or
busy and the timings should be repeated.
"""
import bench_common as bc      # first: one thread for every numerical library

import argparse
import datetime
import os
import subprocess
import sys
import time

JOBS = [
    ['verify_fem_tensor.py'],
    ['verify_fivepoint.py'],
    ['bench_model.py'],
    ['bench_fem2d.py'],
    ['bench_fem2d_cheapest.py'],
    ['bench_fem_quadratic.py'],
    ['bench_fem_bounds_average.py'],
    ['bench_direct_graded.py'],
    ['bench_sor_graded.py'],
    ['bench_sor_squares.py', '--rungs', '1', '2', '3', '4'],
    ['bench_sor_rectangles.py', '--rungs', '1', '2', '3', '4', '5', '6'],
    ['reference_bounds.py'],
    ['plot_benchmarks.py'],
]
LOG = os.path.join(bc.RESULTS_DIR, 'daytime_log.txt')


def say(text, log):
    print(text, flush=True)
    log.write(text + '\n')
    log.flush()


def run(job, log):
    """Run one script in a fresh Python process; copy its output to the screen and the log."""
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, os.path.join(bc.HERE, job[0])] + job[1:], cwd=bc.HERE, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding='utf-8', errors='replace')
    for line in proc.stdout:
        say(line.rstrip('\n'), log)
    proc.wait()
    return proc.returncode, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description='Run the quick benchmarks')
    p.add_argument('--skip-reference', action='store_true', help='leave out reference_bounds.py')
    p.add_argument('--only', nargs='+', help='run only these scripts (file names)')
    a = p.parse_args()
    jobs = [j for j in JOBS if not (a.skip_reference and j[0] == 'reference_bounds.py')]
    if a.only:
        jobs = [j for j in JOBS if j[0] in a.only]
    os.makedirs(bc.RESULTS_DIR, exist_ok=True)
    bc.keep_awake()
    with open(LOG, 'a', encoding='utf-8') as log:
        say(f'\n##### run_daytime.py started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}', log)
        m = bc.machine_info()
        say(f"{m['cpu']} | Python {m['python']}, numpy {m['numpy']}, scipy {m['scipy']}, numba {m['numba']}"
            f" | power throttling off: {bc.POWER_THROTTLING_OFF}", log)
        c0 = bc.calibrate()
        say(f'calibrate(): {c0:.3f} s  (Claude\'s sandbox: 0.95 s)', log)
        failed = []
        for job in jobs:
            say(f"\n----- {' '.join(job)}", log)
            code, t = run(job, log)
            say(f"----- {job[0]} finished in {t:.0f} s{'' if code == 0 else f' WITH ERROR (exit code {code})'}", log)
            if code != 0:
                failed.append(job[0])
        c1 = bc.calibrate()
        say(f'\ncalibrate() at the end: {c1:.3f} s (start {c0:.3f} s)'
            + ('  WARNING: more than 20% slower than at the start -- timings may be affected' if c1 > 1.2 * c0 else ''), log)
        say(f"##### finished {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
            + (f"; FAILED: {', '.join(failed)}" if failed else '; all scripts ran without errors'), log)


if __name__ == '__main__':
    main()
