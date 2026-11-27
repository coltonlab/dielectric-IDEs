"""
run_overnight.py -- run the long SOR benchmarks unattended (one core of a desktop PC, ~8 hours).

    python run_overnight.py                        (default: 8-hour budget)
    python run_overnight.py --budget-hours 6
    python run_overnight.py --c1-rungs 4 5 6 8 --c2-rungs 6 7 8      (choose the rungs)

What it does
  1. Keeps Windows awake (bc.keep_awake) and times bc.calibrate() at the start and at the end; if the
     end is more than 20% slower, it warns of throttling (heat, power plan, other programs).
  2. Collects the jobs: every rung of c1 (bench_sor_squares.py) and c2 (bench_sor_rectangles.py),
     plus any b1 rung (bench_sor_graded.py) estimated to take longer than 15 min (none of the
     default ladder: b1 takes about a minute in all).
  3. Runs them as separate processes, in order of increasing estimated cost.  Estimate of a rung =
     nodes x expected sweeps (deterministic; tables in the scripts) x ns per node-update.  The ns
     come from bc.calibrate() for the first job (4.1 ns in Claude's sandbox, where calibrate takes
     0.95 s), then from the measured speed of the latest (largest) finished job.  A job that would
     not fit in the remaining budget (estimate x 1.1) is skipped; a running job is stopped when the
     budget runs out (its result is then recorded as 'time limit').
  4. Logs everything (the children's output too) to results/overnight_log.txt; results go to
     results/benchmark_results.csv and the S-versus-time trajectories to results/sor_trajectories.csv.
  5. Convergence checks (bench_common.record_check) are not run here: the uniform grids (c1, c2) have
     none (their next grid would have ~93 million nodes), and b1 rungs run here with --no-check because
     b1's check (refine 6, about 1 min) is part of run_daytime.py.  So the budget covers the rungs only.

Estimated run times on our desktop PC if one core is 2x this sandbox (2.05 ns per node-update; the first
job's measurement replaces this guess):
   c1 squares n = 1, 2, 3, 4, 5, 6, 8:  0.1, 0.7, 2.3, 4.0, 7.5, 13, 31 min;  n = 10, 12, 16: 1.0, 1.7, 4.1 h
   c2 2:1     m = 1..8:                 0.03, 0.3, 0.9, 2.1, 4.2, 5.5, 8.3, 12 min;  m = 10, 12, 16: 24, 40, 95 min
   With 8 hours everything except c1 n = 16 fits (about 7 h); c1 n = 16 runs only if the PC is
   faster than assumed.  Memory: 25 bytes per node (c1 n = 16: 9.3 GB).  Ctrl-C stops cleanly.
"""
import bench_common as bc      # first: one thread for every numerical library

import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

import bench_sor_graded as b1
import bench_sor_rectangles as c2
import bench_sor_squares as c1
from bench_direct_graded import LADDER as B1_LADDER

LOG = os.path.join(bc.RESULTS_DIR, 'overnight_log.txt')
C1_RUNGS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
C2_RUNGS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16)
DAYTIME_LIMIT_MIN = 15.0     # b1 rungs estimated to take longer than this run overnight
SAFETY = 1.1                 # a job must fit in the remaining budget with this margin


def log(msg, f):
    line = f'[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
    print(line, flush=True)
    f.write(line + '\n')
    f.flush()


def measured_ns(method, label):
    """ns per node-update of the last SOR run with this method and label (from the trajectory log)."""
    try:
        with open(os.path.join(bc.RESULTS_DIR, 'sor_trajectories.csv'), newline='') as fh:
            rows = [r for r in csv.DictReader(fh) if r['method'] == method and r['label'] == label]
    except OSError:
        return None
    if not rows or int(rows[-1]['sweeps']) == 0:
        return None
    r = rows[-1]
    return 1e9 * float(r['time_s']) / (int(r['sweeps']) * int(r['n_nodes']))


def last_result(method, label):
    try:
        with open(bc.RESULTS_CSV, newline='') as fh:
            rows = [r for r in csv.DictReader(fh) if r['method'] == method and r['label'] == label]
    except OSError:
        return None
    return rows[-1] if rows else None


def main():
    p = argparse.ArgumentParser(description='Run the long SOR benchmarks overnight within a time budget.')
    p.add_argument('--budget-hours', type=float, default=8.0, help='total time budget (default 8)')
    p.add_argument('--c1-rungs', type=float, nargs='*', default=C1_RUNGS, help='c1 rungs n (h = t_e/n)')
    p.add_argument('--c2-rungs', type=float, nargs='*', default=C2_RUNGS, help='c2 rungs m (hy = t_e/m)')
    p.add_argument('--b1-rungs', type=float, nargs='*', default=None,
                   help=f'b1 rungs (default: those estimated over {DAYTIME_LIMIT_MIN:g} min)')
    a = p.parse_args()
    os.makedirs(bc.RESULTS_DIR, exist_ok=True)
    budget = 3600.0 * a.budget_hours
    t0 = time.perf_counter()
    with open(LOG, 'a', encoding='utf-8') as f:
        bc.banner('run_overnight.py -- long SOR benchmarks within a time budget')
        m = bc.machine_info()
        log(f"START  budget {a.budget_hours:g} h  | {m['cpu']} | {m['os']} | Python {m['python']}, numba {m['numba']}", f)
        log(f'keep_awake: {bc.keep_awake()}   power throttling off: {bc.POWER_THROTTLING_OFF}', f)
        cal0 = bc.calibrate()
        ns = c1.SANDBOX_NS_PER_UPDATE * cal0 / c1.SANDBOX_CALIBRATE_S
        log(f'bc.calibrate() = {cal0:.3f} s (sandbox {c1.SANDBOX_CALIBRATE_S} s) -> first guess {ns:.2f} ns per node-update', f)

        # ---- the jobs: (script, rung, method, label, nodes, expected sweeps) ----
        jobs = [('bench_sor_squares.py', n, c1.METHOD, c1.rung_label(n), c1.n_nodes(n), c1.expected_sweeps(n))
                for n in a.c1_rungs]
        jobs += [('bench_sor_rectangles.py', mm, c2.METHOD, c2.rung_label(mm), c2.n_nodes(mm), c2.expected_sweeps(mm))
                 for mm in a.c2_rungs]
        b1_rungs = a.b1_rungs
        if b1_rungs is None:
            b1_rungs = [r for r in B1_LADDER
                        if b1.n_nodes(r) * b1.expected_sweeps(r) * ns * 1e-9 > 60 * DAYTIME_LIMIT_MIN]
        jobs += [('bench_sor_graded.py', r, b1.METHOD, b1.rung_label(r), b1.n_nodes(r), b1.expected_sweeps(r))
                 for r in b1_rungs]
        jobs.sort(key=lambda j: j[4] * j[5])
        log('jobs in order (estimated with the first guess): ' +
            '; '.join(f'{j[0][6:-3]} {j[1]:g}: {j[4] * j[5] * ns * 1e-9 / 60:.1f} min' for j in jobs), f)

        summary = []
        for script, rung, method, label, nodes, sweeps in jobs:
            left = budget - (time.perf_counter() - t0)
            est = nodes * sweeps * ns * 1e-9
            if SAFETY * est > left:
                log(f'SKIP   {script} rung {rung:g} ({label}): estimated {est / 60:.1f} min x {SAFETY} > '
                    f'{left / 60:.1f} min left', f)
                summary.append((script, rung, est, None, 'skipped'))
                continue
            cmd = [sys.executable, '-u', os.path.join(bc.HERE, script), '--rungs', f'{rung:g}',
                   '--budget-hours', f'{left / 3600:.4f}']
            if script != 'bench_sor_graded.py':
                cmd += ['--ns-guess', f'{ns:.4f}']
            else:
                cmd += ['--no-check']      # one rung at a time here; b1's convergence check runs in run_daytime.py
            log(f'RUN    {script} rung {rung:g} ({label}), {nodes:,} nodes x ~{sweeps} sweeps: '
                f'estimated {est / 60:.1f} min, {left / 60:.0f} min left', f)
            t1 = time.perf_counter()
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')   # live, readable log
            proc = subprocess.Popen(cmd, cwd=bc.HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding='utf-8', errors='replace', env=env)
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    f.write('       | ' + line)
                    f.flush()
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                log('INTERRUPTED (Ctrl-C): child stopped; exiting', f)
                break
            took = time.perf_counter() - t1
            got = measured_ns(method, label)
            res = last_result(method, label)
            status = 'failed' if proc.returncode else (
                f"S = {float(res['S']):.7f}, {'beats' if res['beats_model'] == '1' else 'above'} S_MODEL, "
                f"solve {float(res['solve_s']) / 60:.1f} min" if res else 'no result')
            log(f'DONE   {script} rung {rung:g}: {took / 60:.1f} min (estimated {est / 60:.1f}); '
                f"{'' if got is None else f'{got:.2f} ns/update; '}{status}", f)
            summary.append((script, rung, est, took, status))
            if got is not None and not proc.returncode:
                ns = got                       # the latest (largest) job predicts the next best

        cal1 = bc.calibrate()
        log(f'bc.calibrate() at the end = {cal1:.3f} s (start {cal0:.3f} s)', f)
        if cal1 > 1.2 * cal0:
            log('WARNING: the PC was more than 20% slower at the end than at the start (throttling?): '
                'compare ns/update between jobs before trusting the timings', f)
        log('SUMMARY (estimated / actual minutes):', f)
        for script, rung, est, took, status in summary:
            log(f'   {script:24s} {rung:5g}  {est / 60:8.1f} / {"-" if took is None else f"{took / 60:.1f}":>8s}  {status}', f)
        log(f'END    total {(time.perf_counter() - t0) / 3600:.2f} h', f)


if __name__ == '__main__':
    main()
