"""
make_si_data_files.py
=====================

Write the data files of Supplemental section S6 of McClure et al. (2026), one per compound:

    S6_data_MAPbI3.csv, S6_data_PEA2PbI4.csv, S6_data_BA2PbI4.csv, S6_data_HA2PbI4.csv,
    S6_data_PEA2PbBr4.csv, S6_data_PEA2PbCl4.csv

The files published with the paper were made with this script, so running it reproduces them exactly.

    python reproduce_paper.py            (eps_r of every run, about 4 minutes)
    python kramers_kronig.py --paper     (eps_i of every run, about 7 minutes)
    python make_si_data_files.py         (this script: uncertainties, then the files; about 2 minutes)

Output: results/SI_data/.  Each file has one row per frequency (111 Hz to 9.67 MHz) and the columns
  f_Hz               frequency
  eps_r_mean         eps_r of the compound: mean over its films (PEA2PbBr4 and PEA2PbCl4: run 3 only)
  U_total_percent    total uncertainty of eps_r_mean in percent (Sec. IV B; uncertainty.py)
and, for each film,
  <film>_eps_r            eps_r of that film (loss-corrected below 2 MHz, from Cp alone above)
  <film>_eps_i_KK         eps_i from the Kramers-Kronig analysis (Appendix B)
  <film>_eps_i_determined 1 where eps_i is determined by the data (changes by less than 20% when the
                          smoothing or the fitted range is changed), otherwise 0
"""

import csv
import os

import numpy as np

import extract_permittivity as ep
import reproduce_paper as rp
from ide_model import IDEModel, PAPER_MAIN_BATCH, PAPER_SECOND_BATCH
from uncertainty import film_budget, compound_budget

OUT = os.path.join(rp.RESULTS, 'SI_data')
FILE_NAMES = {'MAPbI3': 'S6_data_MAPbI3.csv', 'PEA2PbI4': 'S6_data_PEA2PbI4.csv', 'BA2PbI4': 'S6_data_BA2PbI4.csv',
              'HA2PbI4': 'S6_data_HA2PbI4.csv', 'PEA2PbBr4': 'S6_data_PEA2PbBr4.csv', 'PEA2PbCl4': 'S6_data_PEA2PbCl4.csv'}


def column_prefix(label):
    """Column prefix used in the SI files: 'MAPI s1' -> 'MAPI_s1', 'PEPBr run 3' -> 'PEPBr_r3'."""
    return label.replace(' run ', '_r').replace(' ', '_')


def read_results(path, fmin=100.0, fmax=1e7):
    """Rows of a results CSV (comment lines skipped) between fmin and fmax, as a dict of arrays."""
    with open(path, newline='') as fh:
        rows = [r for r in csv.DictReader(l for l in fh if not l.startswith('#'))]
    rows = [r for r in rows if fmin <= float(r['f_Hz']) <= fmax]
    num = lambda v: float(v) if v not in ('', None) else np.nan
    return {k: np.array([num(r[k]) for r in rows]) for k in rows[0]}


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = {s['label']: s for s in rp.read_samples()}
    geoms = {'main': PAPER_MAIN_BATCH, 'second': PAPER_SECOND_BATCH}
    models = {k: IDEModel(g) for k, g in geoms.items()}
    for comp, runs in rp.COMPOUND_RUNS.items():
        budgets, eps_r, kk = [], {}, {}
        for lab in runs:
            smp = samples[lab]
            base = os.path.join(rp.RESULTS, lab.replace(' ', '_'))
            eps_r[lab] = read_results(base + '.csv')
            kk[lab] = read_results(base + '_KK.csv')
            # uncertainty budget exactly as in uncertainty.py --paper (Cp-only eps_r, as in Sec. IV B)
            t_um = float(smp['thickness_nm']) / 1000.0
            res = ep.extract(os.path.join(rp.DATA, smp['blank_file']), os.path.join(rp.DATA, smp['file']),
                             t_um * 1000, geom=geoms[smp['batch']], loss_cutoff_hz=0.0)
            dt = 100.0 if comp == 'HA2PbI4' else 20.0
            budgets.append(film_budget(models[smp['batch']], res['f_Hz'], res['eps_r_Cp_only'],
                                       res['eps_substrate'], t_um, dt_nm=dt))
            f = res['f_Hz']
            assert np.allclose(f, eps_r[lab]['f_Hz'], rtol=1e-4) and np.allclose(f, kk[lab]['f_Hz'], rtol=1e-4)
        U = compound_budget(budgets)['total']
        mean = np.mean([eps_r[lab]['eps_r'] for lab in runs], axis=0)
        pre = [column_prefix(lab) for lab in runs]
        path = os.path.join(OUT, FILE_NAMES[comp])
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['f_Hz', 'eps_r_mean', 'U_total_percent'] + [p + '_eps_r' for p in pre] +
                       [p + '_eps_i_KK' for p in pre] + [p + '_eps_i_determined' for p in pre])
            for j in range(len(f)):
                w.writerow([f'{f[j]:.6g}', f'{mean[j]:.5g}', f'{U[j]:.3g}'] +
                           [f'{eps_r[lab]["eps_r"][j]:.5g}' for lab in runs] +
                           [f'{kk[lab]["eps_i"][j]:.4g}' for lab in runs] +
                           [int(kk[lab]['eps_i_determined'][j]) for lab in runs])
        print(f'{FILE_NAMES[comp]} written ({len(f)} rows); U_total {np.min(U):.1f}-{np.max(U):.1f}%', flush=True)


if __name__ == '__main__':
    main()
