"""
bench_common.py -- settings and helpers shared by all the benchmark scripts in this folder.

Import it FIRST in every benchmark script, before numpy or scipy:

    import bench_common as bc

because it pins the numerical libraries to ONE thread, so that every method is timed on one core
(SuperLU and the SOR loops are single-threaded anyway; numpy's matrix products would otherwise use
several cores and give the model an unfair advantage).  Set the environment variable
BENCH_THREADS=n to override this for experiments.

What it provides
----------------
* The test problem, identical for every method: the paper's IDE (s = 50.27 um, w = 29.78 um,
  t_e = 115 nm) with a MAPbI3-like film (eps 47, t_p = 437.8 nm from the substrate surface) on quartz
  (eps 4.53).  The grid methods solve the quarter cell 0 < x < a/2, |y| < 3a with V = 0 at y = +-3a
  (moving that boundary to 4a or 5a changes S by about 1e-8 relative; reference_bounds.py).
* S_REF, the best estimate of the true S, and its rigorous bracket S_LOWER <= S_true <= S_UPPER
  (from reference_bounds.py: Dirichlet upper bound and Thomson lower bound on the same mesh).
* S_MODEL, the value of ide_model.py at its default settings: the accuracy target.  Every method in
  this folder except the bound average (d3) gives an UPPER bound on S, so "S <= S_MODEL" means
  "at least as accurate as the model".
* Timing, machine information and a CSV log of results: results/benchmark_results.csv.
* Convergence checks (record_check): each method's cheapest rung at the model's accuracy is solved
  once more with every element size halved, as a user without a reference value would have to do;
  logged as 'check: <rung label>'.
* On Windows: opt this process out of power throttling (EcoQoS), and keep_awake() to stop the
  computer from sleeping during long runs.  Both affect only this Python process while it runs.
"""
import os
import sys

# ---- one thread for every numerical library (must happen before numpy is imported) -----------
_THREADS = os.environ.get('BENCH_THREADS', '1')
if 'numpy' in sys.modules:
    print('bench_common: WARNING numpy was imported before bench_common; thread settings may not apply')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_v] = _THREADS

import csv
import datetime
import platform
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'benchmark_results.csv')

# ---- where fem2d.py and ide_model.py are --------------------------------------------------------
# This folder is benchmarks/ inside the repository, so the repository is its parent folder.  The
# environment variable DIELECTRIC_IDES, if set, points to another copy instead (first match wins).
REPO = None
for _p in (os.environ.get('DIELECTRIC_IDES', ''), os.path.join(HERE, '..')):
    if _p and os.path.isfile(os.path.join(_p, 'ide_model.py')):
        REPO = os.path.abspath(_p)
        sys.path.insert(0, REPO)
        break

# ---- the test problem ---------------------------------------------------------------------------
S_GAP, W_FINGER, T_E = 50.27, 29.78, 0.115     # gap s, finger width w, electrode thickness t_e (um)
A = S_GAP + W_FINGER                           # half-period a = 80.05 um
T_P, EPS_FILM, EPS_SUB = 0.4378, 47.0, 4.53    # film thickness from the substrate (um), eps_film, eps_sub
Y_EXTENT_IN_A = 3.0                            # grid methods: V = 0 at y = +-3a

# ---- reference value and target ------------------------------------------------------------------
# Rigorous bracket of the true S (upper: Dirichlet's principle; lower: Thomson's principle), and the
# best estimate (the midpoint).  From reference_bounds.py: biquadratic elements, upper and lower bound
# on the same 4,714,945-node mesh (our desktop PC, 21 Nov 2026; results/reference.txt), rounded outward at
# the 11th decimal; bracket width 2.0e-10 relative.  (Claude's sandbox, 1,566,025 nodes, had given
# 7.18810231069 <= S <= 7.18810232044, which contains this bracket.)
S_LOWER = 7.18810231472
S_UPPER = 7.18810231618
S_REF = 7.18810231545
S_MODEL = 7.18891898121336       # ide_model.py, default settings: exact Fourier sums (since 21 Nov 2026;
                                 # bench_model.py recomputes it).  The code before that cut the sums off
                                 # after 20,000 terms and gave 7.1889188428426 (1.9e-8 lower).


def rel_err(S):
    """Relative error of S with respect to the best estimate S_REF."""
    return S / S_REF - 1.0


def as_accurate_as_model(S):
    """True if the upper bound S is at least as accurate as the model (S <= S_MODEL, allowing for the
    last digit of S_MODEL)."""
    return S <= S_MODEL * (1.0 + 1e-11)


def err_range(S):
    """Guaranteed range of the relative error of S, from the rigorous bracket."""
    return S / S_UPPER - 1.0, S / S_LOWER - 1.0


# ---- timing -------------------------------------------------------------------------------------
def timed(fn, *args, **kwargs):
    """Run fn once; return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def median_time(fn, repeat=5, *args, **kwargs):
    """Run fn `repeat` times; return (last result, median seconds).  For fast calculations."""
    ts, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def calibrate(n=400, repeat=3):
    """A fixed workload (sparse direct solve of a 400 x 400 grid Laplacian, 160,000 unknowns) used to
    compare machines and to detect throttling during long runs.  Returns the median time in s
    (about 1.0 s in Claude's sandbox, an Intel Xeon at 2.1 GHz)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    K = (sp.kron(sp.eye(n), T) + sp.kron(T, sp.eye(n))).tocsc()
    b = np.ones(n * n)
    _, t = median_time(spla.spsolve, repeat, K, b)
    return t


# ---- machine information ------------------------------------------------------------------------
def cpu_name():
    if sys.platform == 'win32':
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or 'unknown CPU'


def machine_info():
    import scipy
    info = dict(host=platform.node(), cpu=cpu_name(), os=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                threads=_THREADS)
    try:
        import numba
        info['numba'] = numba.__version__
    except ImportError:
        info['numba'] = 'not installed'
    return info


def banner(title):
    """Print a header with the machine and the test problem; call at the start of each script."""
    m = machine_info()
    print('=' * 100)
    print(title)
    print(f"{m['cpu']} | {m['os']} | Python {m['python']}, numpy {m['numpy']}, scipy {m['scipy']}, "
          f"numba {m['numba']} | threads {m['threads']}")
    print(f'Problem: s = {S_GAP} um, w = {W_FINGER} um, t_e = {T_E * 1000:.0f} nm; film eps {EPS_FILM:g}, '
          f't_p = {T_P * 1000:.1f} nm; substrate eps {EPS_SUB}')
    print(f'Reference S = {S_REF:.10f} (true S between {S_LOWER:.11f} and {S_UPPER:.11f});  '
          f'target: model S = {S_MODEL:.7f} ({100 * rel_err(S_MODEL):+.4f}%)')
    print(f"fem2d.py / ide_model.py from: {REPO if REPO else 'NOT FOUND (set DIELECTRIC_IDES)'}")
    print('=' * 100, flush=True)


# ---- results log --------------------------------------------------------------------------------
_FIELDS = ['timestamp', 'host', 'cpu', 'script', 'method', 'label', 'n_nodes', 'setup_s', 'solve_s',
           'total_s', 'per_eval_s', 'S', 'err_pct', 'err_min_pct', 'err_max_pct', 'beats_model', 'notes']


def record(method, label, n_nodes, setup_s, solve_s, S, per_eval_s=None, notes='', quiet=False):
    """Append one result to results/benchmark_results.csv and print it.

    method     : short method name, e.g. 'fem2d (bilinear, direct)'
    label      : which rung of the ladder, e.g. 'h_c = t_e/32, growth 0.06'
    n_nodes    : number of unknowns-plus-fixed nodes (or e.g. Fourier terms for the model; say so in notes)
    setup_s    : time to build the grid/mesh and matrices (s)
    solve_s    : time to solve and compute S (s)
    per_eval_s : time for each further S with new dielectric constants, setup reused (if meaningful)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lo, hi = err_range(S)
    row = dict(timestamp=datetime.datetime.now().isoformat(timespec='seconds'), host=platform.node(),
               cpu=cpu_name(), script=os.path.basename(sys.argv[0]), method=method, label=label,
               n_nodes=n_nodes, setup_s=f'{setup_s:.6g}', solve_s=f'{solve_s:.6g}',
               total_s=f'{setup_s + solve_s:.6g}',
               per_eval_s='' if per_eval_s is None else f'{per_eval_s:.6g}', S=f'{S:.10f}',
               err_pct=f'{100 * rel_err(S):.6f}', err_min_pct=f'{100 * lo:.6f}', err_max_pct=f'{100 * hi:.6f}',
               beats_model=int(as_accurate_as_model(S)), notes=notes)
    new = not os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    if not quiet:
        pe = '' if per_eval_s is None else f', each further S {per_eval_s:.3g} s'
        print(f'{method:32s} {label:34s} {n_nodes:>11,} nodes  setup {setup_s:8.3f} s  solve {solve_s:9.3f} s{pe}'
              f'  S = {S:.7f}  error {100 * rel_err(S):+.4f}%', flush=True)
    return row


# ---- convergence checks -------------------------------------------------------------------------
# Every rung of a ladder is judged against the reference (S_REF), which a user would not have.  To
# learn the error of a mesh without a reference, a user has to solve at least once more on a finer
# mesh.  The benchmarks time the least such check: the same method once more with EVERY element size
# halved (corner size, the other key sizes, the caps and the growth rate), applied to the rung with the
# fewest nodes that is at least as accurate as the model.  The check is logged as its own row, with the
# same method name and the label 'check: <label of the rung it checks>'; plot_benchmarks.py adds its
# time to that rung's first S ('first S + check').  The uniform grids (c1, c2) have no check: their
# next grid would have about 93 million nodes.
CHECK_PREFIX = 'check: '


def check_label(base_label):
    """Label of the row that records the convergence check of the rung `base_label`."""
    return CHECK_PREFIX + base_label


def is_check(label):
    """True for the label of a convergence-check row."""
    return label.startswith(CHECK_PREFIX)


def fewest_nodes(rungs, ok):
    """Of the rungs (dicts with key 'n', the number of nodes) for which ok(rung) is true, the one with
    the fewest nodes; None if there is none."""
    good = [r for r in rungs if ok(r)]
    return min(good, key=lambda r: r['n']) if good else None


def record_check(method, base_label, S_base, n_nodes, setup_s, solve_s, S,
                 what='all element sizes / 2', extra_notes=''):
    """Log the convergence check of the rung `base_label` (whose S was S_base): n_nodes, setup_s,
    solve_s and S are those of the check mesh.  The notes give the check's estimate of the rung's
    error, S_base/S_check - 1 (what a user would see), and the rung's true error (from S_REF).  Prints
    one line and returns the estimate."""
    est = S_base / S - 1.0
    notes = f'{what}; estimate S_base/S_check - 1 = {est:+.3e}; true error of base = {rel_err(S_base):+.3e}'
    if extra_notes:
        notes += '; ' + extra_notes
    record(method, check_label(base_label), n_nodes, setup_s, solve_s, S, per_eval_s=solve_s, notes=notes,
           quiet=True)
    print(f'CHECK of [{base_label}] ({what}): {n_nodes:,} nodes, S = {S:.10f}, setup {setup_s:.4f} s, '
          f'solve {solve_s:.4f} s;  estimated error of the rung {est:+.2e}, true {rel_err(S_base):+.2e}',
          flush=True)
    return est


# ---- Windows: no power throttling, no sleep -----------------------------------------------------
def _no_power_throttling():
    """Ask Windows not to run this process in its power-saving mode (EcoQoS), which can slow a process
    whose window is in the background and move it to the slower E-cores.  Documented Win32 call
    SetProcessInformation(ProcessPowerThrottling); affects only this process."""
    if sys.platform != 'win32':
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _PPTS(ctypes.Structure):
            _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG),
                        ('StateMask', wintypes.ULONG)]
        state = _PPTS(1, 0x1, 0x0)       # version 1; control EXECUTION_SPEED; state 0 = never throttle
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        # The argument types must be declared: without them ctypes tries to pass the process handle (the
        # pseudo-handle -1, returned as 2**64 - 1) as a 32-bit int, raises OverflowError, and the call
        # fails silently -- as in the runs of 21 Nov 2026 ('power throttling off: False').  Fixed 22 Nov.
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4,   # 4 = ProcessPowerThrottling
                                              ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:
        return False


POWER_THROTTLING_OFF = _no_power_throttling()


def keep_awake():
    """Stop Windows from going to sleep while this process runs (SetThreadExecutionState with
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED).  The setting ends when the process exits."""
    if sys.platform != 'win32':
        return False
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001))
    except Exception:
        return False
