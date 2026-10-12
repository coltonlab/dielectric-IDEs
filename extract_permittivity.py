"""
extract_permittivity.py
=======================

Calculate the dielectric function of a thin film from impedance measurements of
  (1) a BLANK IDE (bare substrate, no film), and
  (2) an IDE of the same design with the FILM on it,
using the model in ide_model.py.

This is the script most users want.


INPUT DATA FILES
----------------
Plain-text CSV files, one row per frequency, no header needed (lines that do not start with a
number are skipped).  By default the columns are those written by our measurement program:

    column 0 : frequency f (Hz)
    column 1 : admittance magnitude |Y| (S)     <- not used
    column 2 : parallel conductance G (S)
    column 3 : parallel capacitance Cp (F)

i.e. the analyzer (an HP 4192A) in parallel-equivalent-circuit mode.  If your files are laid out
differently, give the column numbers (counting from 0) with --columns f,G,Cp, for example
--columns 0,2,1.  If you have no conductance column, use -1 for G (--columns 0,-1,1); the
loss correction is then skipped and only the Cp-based result is computed.

The blank and film files do not need the same frequencies; the substrate result is interpolated
(linearly in log f) onto the film's frequencies if they differ.


WHAT IS CALCULATED
------------------
1. From the blank: the substrate dielectric constant eps_Q at every frequency (Eq. 13).
2. From the film: the film dielectric constant eps_P, given eps_Q and the film thickness, two ways:
   a. from Cp alone (eps_r_Cp_only), and
   b. below a cutoff frequency (default 2 MHz), from the complex capacitance Cp - iG/omega with
      complex dielectric constants (eps_r_loss_corrected and eps_loss_incl_conduction).
      Fitting Cp alone overestimates eps' of a lossy film (Sec. III B 7 of the paper), so this is
      the better value where G is measured accurately.  The paper uses it below 2 MHz; above
      2 MHz our measured G had a setup-dependent error, so there we use Cp alone.
   The column eps_r combines the two the way the paper does.
   NOTE: eps_loss_incl_conduction includes DC conduction and any error in the measured G.  The
   paper's imaginary part eps_i comes instead from a Kramers-Kronig analysis of eps_r
   (kramers_kronig.py), which excludes conduction.


HOW TO RUN IT (examples use the data files of the paper, in the data/ folder)
---------------------------------------------------------------------------
Film on our IDEs (MAPbI3 film s1, 437.8 nm thick):

    python extract_permittivity.py --blank data/blank_main_batch.csv \
        --film data/MAPbI3_s1.csv --thickness-nm 437.8 --out MAPI_s1.csv --plot

Substrate only (no --film):

    python extract_permittivity.py --blank data/blank_main_batch.csv --out quartz.csv

A liquid covering the electrodes, e.g. water (--bulk sets the layer thickness to 1 cm):

    python extract_permittivity.py --blank data/blank_main_batch.csv \
        --film data/water_1cm.csv --bulk --no-loss-correction --out water.csv

Your own electrodes (lengths in um, overlap length in mm):

    python extract_permittivity.py --blank myblank.csv --film myfilm.csv --thickness-nm 250 \
        --s-um 10 --w-um 10 --te-um 0.1 --N 50 --L-mm 2.0

Run  python extract_permittivity.py --help  for all options.

Choosing the electrode dimensions matters: the result depends on the gap s, width w, electrode
thickness t_e, number of fingers N and overlap length L (Sec. III B 2), and most of all on the
film thickness.  Measure them on your own substrates.  See uncertainty.py for how errors in
these inputs propagate to eps_r.
"""

import argparse
import csv
import os
import sys

import numpy as np

from ide_model import IDEGeometry, IDEModel, PAPER_MAIN_BATCH, PAPER_SECOND_BATCH

BULK_THICKNESS_UM = 1.0e4      # 1 cm: "infinitely" thick compared with the electrode period


# --------------------------------------------------------------------------------------------
# Reading data files
# --------------------------------------------------------------------------------------------
def read_impedance_file(path, columns=(0, 2, 3)):
    """Read a measurement file.  Returns (f, G, Cp) as numpy arrays (G is None if the G column
    is given as -1).  Lines that do not start with a number (headers, comments) are skipped."""
    icol_f, icol_G, icol_C = columns
    rows = []
    with open(path, newline='') as fh:
        for line in fh:
            parts = [p.strip() for p in line.replace(';', ',').replace('\t', ',').split(',')]
            try:
                rows.append([float(p) for p in parts if p != ''])
            except ValueError:
                continue                    # header or comment line
    if not rows:
        raise ValueError(f'No numeric data found in {path}')
    data = np.array(rows)
    f = data[:, icol_f]
    Cp = data[:, icol_C]
    G = data[:, icol_G] if icol_G >= 0 else None
    order = np.argsort(f)                   # make sure frequencies increase
    return f[order], (G[order] if G is not None else None), Cp[order]


def _interp_logf(f_new, f_old, values, rtol=1e-4):
    """Values of the substrate (computed at the blank's frequencies f_old) at the film's
    frequencies f_new.  A film frequency within a relative 1e-4 of a blank frequency is treated as
    the same frequency (files often store the frequencies with different numbers of digits);
    otherwise the value is interpolated linearly in log f (NaN if a neighbor is NaN)."""
    v = np.asarray(values)
    if f_new.min() < f_old.min() * 0.999 or f_new.max() > f_old.max() * 1.001:
        print('WARNING: the film data extend beyond the frequency range of the blank; the substrate '
              'values are extrapolated as constants there.', file=sys.stderr)
    lx, lo = np.log(f_new), np.log(f_old)
    if np.iscomplexobj(v):
        out = np.interp(lx, lo, v.real) + 1j * np.interp(lx, lo, v.imag)
    else:
        out = np.interp(lx, lo, v)
    nearest = np.argmin(np.abs(lx[:, None] - lo[None, :]), axis=1)
    same = np.abs(f_old[nearest] / f_new - 1) < rtol
    out[same] = v[nearest[same]]
    return out


# --------------------------------------------------------------------------------------------
# The calculation, usable from other scripts
# --------------------------------------------------------------------------------------------
def substrate_spectrum(model, f, G, Cp, loss_cutoff_hz=2e6):
    """Substrate dielectric constant at every frequency in f.
    Returns a dict with 'eps_Cp' (real, from Cp alone) and 'eps_complex' (complex, from
    Cp - iG/omega below loss_cutoff_hz; NaN above it or if G is None)."""
    eps_cp = np.array([model.solve_substrate(c) for c in Cp])
    eps_cx = np.full(len(f), np.nan + 0j)
    if G is not None and loss_cutoff_hz > 0:
        for i in np.where(f < loss_cutoff_hz)[0]:
            try:
                eps_cx[i] = model.solve_substrate_complex(Cp[i], G[i], f[i], eps_start=eps_cp[i])
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass
    return dict(eps_Cp=eps_cp, eps_complex=eps_cx)


def film_spectrum(model, f, G, Cp, eps_sub_cp, eps_sub_complex, t_film_um, loss_cutoff_hz=2e6):
    """Film dielectric function at every frequency in f.

    eps_sub_cp, eps_sub_complex : substrate values at the same frequencies (from substrate_spectrum).
    Returns a dict of arrays:
      eps_r_Cp_only            eps' from Cp alone
      eps_complex              eps' - i eps'' from Cp - iG/omega (below the cutoff, else NaN)
      eps_r                    the paper's choice: Re(eps_complex) below the cutoff, eps_r_Cp_only above
    NaN marks frequencies where no solution exists (e.g. film capacitance below the blank's)."""
    n = len(f)
    eps_cp = np.full(n, np.nan)
    eps_cx = np.full(n, np.nan + 0j)
    for i in range(n):
        try:
            eps_cp[i] = model.solve_film(Cp[i], eps_sub_cp[i], t_film_um)
        except ValueError:
            pass
    # Complex fit, from high to low frequency.  Newton's method starts from the Cp-only value (with a
    # small loss); if that fails it starts from the solution at the next higher frequency instead.
    previous = None
    for i in sorted(range(n), key=lambda j: -f[j]):
        if G is None or f[i] >= loss_cutoff_hz or not np.isfinite(eps_sub_complex[i]):
            continue
        starts = ([eps_cp[i] - 0.1j] if np.isfinite(eps_cp[i]) else []) + ([previous] if previous is not None else [])
        for x0 in starts:
            try:
                eps_cx[i] = model.solve_film_complex(Cp[i], G[i], f[i], eps_sub_complex[i], t_film_um, eps_start=x0)
                previous = eps_cx[i]
                break
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
    use_cx = (f < loss_cutoff_hz) & np.isfinite(eps_cx)
    eps_r = np.where(use_cx, eps_cx.real, eps_cp)
    return dict(eps_r_Cp_only=eps_cp, eps_complex=eps_cx, eps_r=eps_r)


def extract(blank_file, film_file=None, thickness_nm=None, geom=None, fmin=100.0, fmax=1e7,
            loss_cutoff_hz=2e6, columns=(0, 2, 3), bulk=False, substrate_fixed_below_hz=None):
    """Everything in one call: read the files, compute the substrate and (optionally) film spectra
    between fmin and fmax.  Returns a dict of arrays (see write_csv for the meaning of each).

    substrate_fixed_below_hz : optional.  Below this frequency the blank's own data are not used:
        the substrate is taken as lossless, with the eps it has at the lowest blank frequency at or
        above substrate_fixed_below_hz.  Use it to analyze a film below the frequency where the
        blank's capacitance becomes unreliable (for our blanks, below about 100 Hz; the paper uses
        this for the Kramers-Kronig input of its thickest films, Appendix B, with a value of 90 Hz)."""
    geom = geom or IDEGeometry()
    model = IDEModel(geom)
    fb, Gb, Cb = read_impedance_file(blank_file, columns)
    sub = substrate_spectrum(model, fb, Gb, Cb, loss_cutoff_hz)
    if substrate_fixed_below_hz:
        i0 = int(np.argmax(fb >= substrate_fixed_below_hz))      # first blank frequency at or above the limit
        low = fb < fb[i0]
        sub['eps_Cp'][low] = sub['eps_Cp'][i0]
        sub['eps_complex'][low] = sub['eps_Cp'][i0] + 0j
    out = dict(geometry=geom)
    if film_file is None:
        keep = (fb >= fmin) & (fb <= fmax)
        out.update(f_Hz=fb[keep], eps_substrate=sub['eps_Cp'][keep],
                   eps_substrate_complex=sub['eps_complex'][keep])
        return out
    if bulk:
        t_um = BULK_THICKNESS_UM
    elif thickness_nm is None:
        raise ValueError('Give the film thickness (thickness_nm), or bulk=True for a liquid.')
    else:
        t_um = thickness_nm / 1000.0
    ff, Gf, Cf = read_impedance_file(film_file, columns)
    keep = (ff >= fmin) & (ff <= fmax)
    ff, Cf = ff[keep], Cf[keep]
    Gf = Gf[keep] if Gf is not None else None
    eq_cp = _interp_logf(ff, fb, sub['eps_Cp'])
    eq_cx = _interp_logf(ff, fb, sub['eps_complex'])
    film = film_spectrum(model, ff, Gf, Cf, eq_cp, eq_cx, t_um, loss_cutoff_hz)
    out.update(f_Hz=ff, eps_substrate=eq_cp, eps_substrate_complex=eq_cx, thickness_um=t_um, **film)
    return out


def write_csv(path, res, header_lines=()):
    """Write the results to a CSV file, with a commented header describing the columns."""
    g = res['geometry']
    with open(path, 'w', newline='') as fh:
        for line in header_lines:
            fh.write('# ' + line + '\n')
        fh.write(f'# geometry: s = {g.s_um} um, w = {g.w_um} um, t_e = {g.te_um} um, N = {g.N}, '
                 f'L = {g.L_mm} mm, N_eff = {g.n_eff():.4f}\n')
        if 'eps_r' not in res:
            fh.write('# columns: f_Hz; eps_substrate = real eps of the substrate from Cp; '
                     'eps_substrate_re/_loss = eps\', eps\'\' of the substrate from Cp - iG/omega\n')
            w = csv.writer(fh)
            w.writerow(['f_Hz', 'eps_substrate', 'eps_substrate_re', 'eps_substrate_loss'])
            for i in range(len(res['f_Hz'])):
                z = res['eps_substrate_complex'][i]
                w.writerow([f'{res["f_Hz"][i]:.6g}', f'{res["eps_substrate"][i]:.6g}',
                            _fmt(z.real), (_fmt(-z.imag) if np.isfinite(z) else '')])
            return
        fh.write(f'# film thickness = {res["thickness_um"] * 1000:.6g} nm\n')
        fh.write('# columns: f_Hz; eps_substrate (from the blank, Cp); eps_r_Cp_only (film eps\' from Cp alone); '
                 'eps_r_loss_corrected and eps_loss_incl_conduction (film eps\' and eps\'\' from Cp - iG/omega, '
                 'below the loss-correction cutoff; eps\'\' includes DC conduction); eps_r (recommended eps\')\n')
        w = csv.writer(fh)
        w.writerow(['f_Hz', 'eps_substrate', 'eps_r_Cp_only', 'eps_r_loss_corrected',
                    'eps_loss_incl_conduction', 'eps_r'])
        for i in range(len(res['f_Hz'])):
            z = res['eps_complex'][i]
            w.writerow([f'{res["f_Hz"][i]:.6g}', f'{res["eps_substrate"][i]:.6g}', _fmt(res['eps_r_Cp_only'][i]),
                        _fmt(z.real), (_fmt(-z.imag) if np.isfinite(z) else ''), _fmt(res['eps_r'][i])])


def _fmt(x):
    return f'{x:.6g}' if np.isfinite(x) else ''


def plot_results(path_png, res, title=''):
    """Save a quick log-log plot of eps' (and eps'' from G, if computed) versus frequency."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    f = res['f_Hz']
    if 'eps_r' in res:
        ax.loglog(f, res['eps_r'], 'k-', label="eps' (recommended)")
        ax.loglog(f, res['eps_r_Cp_only'], 'k:', label="eps' from Cp alone")
        loss = -res['eps_complex'].imag
        if np.any(np.isfinite(loss) & (loss > 0)):
            ax.loglog(f, np.where(loss > 0, loss, np.nan), 'r--', label="eps'' from G (incl. conduction)")
    else:
        ax.semilogx(f, res['eps_substrate'], 'k-', label='substrate eps')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Dielectric function')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path_png)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description='Dielectric function of a thin film on interdigitated electrodes '
                                            '(McClure et al. 2026). See the top of this file for examples.')
    p.add_argument('--blank', required=True, help='data file of the blank IDE (bare substrate)')
    p.add_argument('--film', help='data file of the IDE with the film; omit to compute only the substrate')
    p.add_argument('--thickness-nm', type=float, help='film thickness in nm, measured from the substrate surface')
    p.add_argument('--bulk', action='store_true', help='the "film" is a thick layer (e.g. a liquid) covering the '
                                                      'electrodes; its thickness is set to 1 cm')
    p.add_argument('--batch', choices=['main', 'second'], default='main',
                   help="electrode dimensions of the paper's IDEs: main wafer batch (default) or second batch")
    p.add_argument('--s-um', type=float, help='gap between electrodes (um); overrides --batch')
    p.add_argument('--w-um', type=float, help='electrode width (um)')
    p.add_argument('--te-um', type=float, help='electrode thickness (um)')
    p.add_argument('--N', type=int, help='number of fingers')
    p.add_argument('--L-mm', type=float, help='overlap length of adjacent fingers (mm)')
    p.add_argument('--N-eff', type=float, help='effective number of half-periods (default: N - 1/2 + N a/L)')
    p.add_argument('--fmin', type=float, default=100.0, help='lowest frequency to analyze, Hz (default 100)')
    p.add_argument('--fmax', type=float, default=1e7, help='highest frequency to analyze, Hz (default 1e7)')
    p.add_argument('--loss-cutoff', type=float, default=2e6,
                   help='use the loss-corrected (complex) fit below this frequency, Hz (default 2e6)')
    p.add_argument('--no-loss-correction', action='store_true', help='use Cp alone at all frequencies')
    p.add_argument('--substrate-fixed-below', type=float, default=None,
                   help='below this frequency (Hz) use a lossless substrate with its eps at the first blank '
                        'frequency at or above it, instead of the blank data (see extract())')
    p.add_argument('--columns', default='0,2,3', help='column numbers (from 0) of f, G and Cp; '
                                                       'use -1 for G if there is none (default 0,2,3)')
    p.add_argument('--out', help='output CSV file (default: <film or blank name>_eps.csv)')
    p.add_argument('--plot', action='store_true', help='also save a PNG plot next to the CSV file')
    a = p.parse_args(argv)

    base = PAPER_SECOND_BATCH if a.batch == 'second' else PAPER_MAIN_BATCH
    geom = IDEGeometry(s_um=a.s_um if a.s_um is not None else base.s_um,
                       w_um=a.w_um if a.w_um is not None else base.w_um,
                       te_um=a.te_um if a.te_um is not None else base.te_um,
                       N=a.N if a.N is not None else base.N,
                       L_mm=a.L_mm if a.L_mm is not None else base.L_mm,
                       N_eff=a.N_eff)
    cols = tuple(int(c) for c in a.columns.split(','))
    cutoff = 0.0 if a.no_loss_correction else a.loss_cutoff
    if a.film and a.thickness_nm is None and not a.bulk:
        p.error('give --thickness-nm (or --bulk for a thick liquid layer)')

    res = extract(a.blank, a.film, a.thickness_nm, geom, a.fmin, a.fmax, cutoff, cols, a.bulk,
                  a.substrate_fixed_below)
    out = a.out or (os.path.splitext(os.path.basename(a.film or a.blank))[0] + '_eps.csv')
    write_csv(out, res, header_lines=[f'blank file: {a.blank}', f'film file: {a.film}',
                                      f'loss-correction cutoff: {cutoff:g} Hz'])
    if a.plot:
        plot_results(os.path.splitext(out)[0] + '.png', res, title=os.path.basename(a.film or a.blank))

    # short summary on the screen
    f = res['f_Hz']
    print(f'N_eff = {geom.n_eff():.3f};  results written to {out}')
    for target in (1e3, 1e5, 1.06e6, 9.67e6):
        if target < f.min() / 1.2 or target > f.max() * 1.2:
            continue
        i = int(np.argmin(np.abs(np.log(f / target))))
        line = f'  f = {f[i]:12.1f} Hz   eps_substrate = {res["eps_substrate"][i]:.4f}'
        if 'eps_r' in res:
            line += f'   film eps_r = {res["eps_r"][i]:.4g}'
            z = res['eps_complex'][i]
            if np.isfinite(z):
                line += f'   (Cp only {res["eps_r_Cp_only"][i]:.4g}; eps\'\' from G {-z.imag:.3g})'
        print(line)


if __name__ == '__main__':
    main()
