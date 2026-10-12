"""
uncertainty.py
==============

Uncertainty of a film's dielectric constant eps_r, item by item, at every frequency (Sec. IV B of
McClure et al. 2026).

    python uncertainty.py --blank data/blank_main_batch.csv \
        --film data/MAPbI3_s1.csv --thickness-nm 437.8 --out MAPI_s1_unc.csv

    python uncertainty.py --paper        (all compounds of the paper, about 2 minutes)


THE ITEMS (all in percent of eps_r; combined in quadrature)
-----------------------------------------------------------
 (i)   absolute film thickness, +-20 nm by default (--dt-nm).  Systematic.
 (ii)  film-to-film thickness variation within a batch, 4.4% of the thickness by default
       (--dt-rel 0.044); needed when, as for us, the thickness was measured on only one or two
       films of each batch.  Random.
 (iii) chip-to-chip variation of the blank capacitance, 0.3% by default (--blank-rel 0.003).
       Each film is measured on its own chip but compared with the blank of another chip.  Random.
 (iv)  instrumental (setup-dependent) capacitance error at high frequency: 0.27% at 9.67 MHz by
       default (--hf-rel 0.0027 at --hf-freq 9.6748e6, our highest frequency), falling off as f^2.  Like item (iii) it is
       propagated as an error in the blank capacitance relative to the film's.  Random.
 (v)   electrode geometry and model, 0.5% of eps_r by default (--model-rel 0.005).  Systematic.
 (vi)  capacitance resolution of the analyzer, 0.0041 pF by default (--resolution-pF).  Random.

The thickness items (i, ii) usually dominate.  Set each item to match YOUR measurements: the
defaults are the values we used for our films and analyzer.

HOW EACH ITEM IS PROPAGATED
---------------------------
All items are small, so they are propagated linearly.  With S(eps_P, eps_Q, t) the model's
dimensionless capacitance (ide_model.py), the sensitivities are obtained by finite differences:

  thickness:   d eps_P = -(dS/dt) / (dS/deps_P) * dt
  blank error: a fractional error x in the blank capacitance shifts the substrate value by
               d eps_Q = x * S_blank / (dS_blank/deps_Q), which shifts the film value by
               d eps_P = -(dS_film/deps_Q) / (dS_film/deps_P) * d eps_Q
  resolution:  d eps_P = (dC / prefactor) / (dS/deps_P)

Because eps_P comes from the SMALL difference between the film and blank capacitances, errors in
the blank are amplified for thin films and for films whose eps_r is close to the substrate's
(e.g. our 360 nm PEA2PbCl4 film).

COMPOUND VALUES
---------------
For a compound measured on n films, the random items (ii, iii, iv, vi) are divided by sqrt(n);
the systematic items (i, v) are not.  The paper's PEA2PbBr4 and PEA2PbCl4 values are each one film
(run 3; measured twice), so n = 1 for them.

The eps values used for the sensitivities are those from Cp alone (eps_r_Cp_only); using the
loss-corrected values instead changes the percentages very little.
"""

import argparse
import csv
import os

import numpy as np

import extract_permittivity as ep
from ide_model import IDEModel, PAPER_MAIN_BATCH, PAPER_SECOND_BATCH

ITEMS = ['thickness_abs', 'thickness_film_to_film', 'blank_chip_to_chip', 'hf_instrument',
         'geometry_model', 'resolution']
RANDOM_ITEMS = ['thickness_film_to_film', 'blank_chip_to_chip', 'hf_instrument', 'resolution']


def film_budget(model, f, eps_film, eps_sub, t_um, dt_nm=20.0, dt_rel=0.044, blank_rel=0.003,
                hf_rel=0.0027, hf_freq=9.6748e6, model_rel=0.005, resolution_pF=0.0041):
    """Percent uncertainty of eps_film for each item, at every frequency.

    model    : IDEModel for the electrodes
    f        : frequencies (Hz);  eps_film, eps_sub : film and substrate eps at those frequencies
    t_um     : film thickness (um)
    Returns a dict {item: array of percent values, 'total': array}.
    """
    S = model.S
    te = model.te
    out = {k: np.full(len(f), np.nan) for k in ITEMS + ['total']}
    for j, (e, q) in enumerate(zip(eps_film, eps_sub)):
        if not np.isfinite(e):
            continue
        h = 1e-4
        dS_de = (S(e * (1 + h), q, t_um) - S(e * (1 - h), q, t_um)) / (2 * h * e)    # dS/d eps_film
        dS_dq = (S(e, q + h, t_um) - S(e, q - h, t_um)) / (2 * h)                    # dS/d eps_sub (film)
        dS_dt = (S(e, q, t_um + h) - S(e, q, t_um - h)) / (2 * h)                    # dS/d t (per um)
        dSb_dq = (model.S_blank(q + h) - model.S_blank(q - h)) / (2 * h)             # dS_blank/d eps_sub
        Sb = model.S_blank(q)

        pct = lambda d_eps: abs(d_eps) / e * 100.0
        out['thickness_abs'][j] = pct(dS_dt * (dt_nm / 1000.0) / dS_de)                  # item (i)
        out['thickness_film_to_film'][j] = pct(dS_dt * dt_rel * t_um / dS_de)            # item (ii)
        per_percent_blank = pct(dS_dq * (0.01 * Sb / dSb_dq) / dS_de)                    # % in eps per 1% in C_blank
        out['blank_chip_to_chip'][j] = per_percent_blank * (100 * blank_rel)             # item (iii)
        out['hf_instrument'][j] = per_percent_blank * (100 * hf_rel) * (f[j] / hf_freq) ** 2   # item (iv)
        out['geometry_model'][j] = 100 * model_rel                                       # item (v)
        out['resolution'][j] = pct(resolution_pF * 1e-12 / model.geom.prefactor() / dS_de)    # item (vi)
    out['total'] = np.sqrt(sum(out[k] ** 2 for k in ITEMS))
    return out


def compound_budget(budgets, n_films=None, model_rel=0.005):
    """Combine the budgets of several films (or runs) of one compound.  Systematic items are averaged;
    random items are averaged in quadrature and divided by sqrt(n_films) (default: number of budgets)."""
    n = n_films or len(budgets)
    absol = np.nanmean([b['thickness_abs'] for b in budgets], axis=0)
    rnd = np.sqrt(np.nanmean([sum(b[k] ** 2 for k in RANDOM_ITEMS) for b in budgets], axis=0)) / np.sqrt(n)
    total = np.sqrt(absol ** 2 + rnd ** 2 + (100 * model_rel) ** 2)
    return dict(systematic_thickness=absol, random=rnd, total=total)


def write_budget_csv(path, f, eps, budget):
    with open(path, 'w', newline='') as fh:
        fh.write('# percent uncertainty of eps_r by item (Sec. IV B); total = quadrature sum\n')
        w = csv.writer(fh)
        w.writerow(['f_Hz', 'eps_r_Cp_only'] + [k + '_pct' for k in ITEMS] + ['total_pct'])
        for j in range(len(f)):
            w.writerow([f'{f[j]:.6g}', f'{eps[j]:.6g}'] + [f'{budget[k][j]:.4g}' for k in ITEMS] +
                       [f'{budget["total"][j]:.4g}'])


def paper(dt_nm=20.0):
    """Reproduce the uncertainties of Sec. IV B for all films and compounds of the paper."""
    import reproduce_paper as rp
    geoms = {'main': PAPER_MAIN_BATCH, 'second': PAPER_SECOND_BATCH}
    models = {k: IDEModel(g) for k, g in geoms.items()}
    films, eps = {}, {}
    for smp in rp.read_samples():
        if smp['thickness_nm'] == 'bulk':
            continue
        t_um = float(smp['thickness_nm']) / 1000.0
        res = ep.extract(os.path.join(rp.DATA, smp['blank_file']), os.path.join(rp.DATA, smp['file']), t_um * 1000,
                         geom=geoms[smp['batch']], loss_cutoff_hz=0.0)       # Cp only: fast
        dt = 100.0 if smp['compound'] == 'HA2PbI4' else dt_nm               # HA2PbI4: +-100 nm (Sec. IV B)
        films[smp['label']] = film_budget(models[smp['batch']], res['f_Hz'], res['eps_r_Cp_only'],
                                          res['eps_substrate'], t_um, dt_nm=dt)
        eps[smp['label']] = res['eps_r_Cp_only']
        f = res['f_Hz']
        print(f'{smp["label"]} done', flush=True)
    j = int(np.argmin(np.abs(np.log(f / 9.67e6))))
    print(f'\nPer-film uncertainty (%) at {f[j] / 1e6:.2f} MHz: items (i)-(vi) and total')
    for lab, b in films.items():
        print(f'  {lab:12s} ' + ' '.join(f'{b[k][j]:5.2f}' for k in ITEMS) + f'   total {b["total"][j]:5.2f}'
              f'   eps_r = {eps[lab][j]:.2f} +- {eps[lab][j] * b["total"][j] / 100:.2f}')
    print('\nCompound values at 9.67 MHz (paper, Sec. IV B: 47.3 +- 2.5, 10.5 +- 0.4, 10.6 +- 0.6, 9.3 +- 0.7, '
          '6.1 +- 0.3, 6.0 +- 0.9):')
    for comp, runs in rp.COMPOUND_RUNS.items():
        cb = compound_budget([films[r] for r in runs])
        mean = np.mean([eps[r][j] for r in runs])
        u = cb['total']
        print(f'  {comp:10s} eps_r = {mean:6.2f} +- {mean * u[j] / 100:.2f}  ({u[j]:.1f}%; '
              f'{np.nanmin(u):.1f}-{np.nanmax(u):.1f}% over 111 Hz-9.67 MHz)')


def main(argv=None):
    p = argparse.ArgumentParser(description='Uncertainty budget of eps_r (Sec. IV B of McClure et al. 2026).')
    p.add_argument('--paper', action='store_true', help='reproduce the uncertainties of the paper')
    p.add_argument('--blank', help='data file of the blank IDE')
    p.add_argument('--film', help='data file of the IDE with the film')
    p.add_argument('--thickness-nm', type=float, help='film thickness (nm)')
    p.add_argument('--batch', choices=['main', 'second'], default='main', help="the paper's electrodes (default main)")
    p.add_argument('--s-um', type=float); p.add_argument('--w-um', type=float); p.add_argument('--te-um', type=float)
    p.add_argument('--N', type=int); p.add_argument('--L-mm', type=float); p.add_argument('--N-eff', type=float)
    p.add_argument('--columns', default='0,2,3', help='column numbers of f, G, Cp (default 0,2,3)')
    p.add_argument('--fmin', type=float, default=100.0); p.add_argument('--fmax', type=float, default=1e7)
    p.add_argument('--dt-nm', type=float, default=20.0, help='(i) absolute thickness uncertainty, nm (default 20)')
    p.add_argument('--dt-rel', type=float, default=0.044, help='(ii) film-to-film thickness variation, fraction (default 0.044)')
    p.add_argument('--blank-rel', type=float, default=0.003, help='(iii) chip-to-chip blank capacitance variation, fraction (default 0.003)')
    p.add_argument('--hf-rel', type=float, default=0.0027, help='(iv) instrumental capacitance error at --hf-freq, fraction (default 0.0027)')
    p.add_argument('--hf-freq', type=float, default=9.6748e6, help='frequency at which --hf-rel applies, Hz (default 9.6748e6, our highest)')
    p.add_argument('--model-rel', type=float, default=0.005, help='(v) geometry and model, fraction of eps (default 0.005)')
    p.add_argument('--resolution-pF', type=float, default=0.0041, help='(vi) capacitance resolution, pF (default 0.0041)')
    p.add_argument('--out', help='output CSV (default <film name>_uncertainty.csv)')
    a = p.parse_args(argv)
    if a.paper:
        paper(a.dt_nm)
        return
    if not (a.blank and a.film and a.thickness_nm):
        p.error('give --blank, --film and --thickness-nm (or --paper)')
    base = PAPER_SECOND_BATCH if a.batch == 'second' else PAPER_MAIN_BATCH
    from ide_model import IDEGeometry
    geom = IDEGeometry(s_um=a.s_um or base.s_um, w_um=a.w_um or base.w_um,
                       te_um=a.te_um if a.te_um is not None else base.te_um, N=a.N or base.N,
                       L_mm=a.L_mm or base.L_mm, N_eff=a.N_eff)
    cols = tuple(int(c) for c in a.columns.split(','))
    res = ep.extract(a.blank, a.film, a.thickness_nm, geom, a.fmin, a.fmax, 0.0, cols)
    b = film_budget(IDEModel(geom), res['f_Hz'], res['eps_r_Cp_only'], res['eps_substrate'], a.thickness_nm / 1000,
                    a.dt_nm, a.dt_rel, a.blank_rel, a.hf_rel, a.hf_freq, a.model_rel, a.resolution_pF)
    out = a.out or os.path.splitext(os.path.basename(a.film))[0] + '_uncertainty.csv'
    write_budget_csv(out, res['f_Hz'], res['eps_r_Cp_only'], b)
    f = res['f_Hz']
    print(f'results written to {out}')
    for target in (1e3, 1e5, 1.06e6, 9.67e6):
        j = int(np.argmin(np.abs(np.log(f / target))))
        print(f'  f = {f[j]:12.1f} Hz  eps_r = {res["eps_r_Cp_only"][j]:.4g}  total {b["total"][j]:.2f}%  ('
              + ', '.join(f'{k} {b[k][j]:.2f}' for k in ITEMS) + ')')


if __name__ == '__main__':
    main()
