"""
reproduce_paper.py
==================

Recompute the dielectric constants reported in McClure et al. (2026) from the raw data in data/,
and compare them with the values printed in the paper.

    python reproduce_paper.py            (takes about 4-5 minutes)

What it does:
  1. Reads data/samples.csv, which lists every data file with its film thickness and wafer batch.
  2. For each wafer batch, computes the quartz dielectric constant from the blank (Sec. 4.1).
  3. For each film run, computes eps_r(f) between 100 Hz and 10 MHz exactly as in the paper:
     loss-corrected (complex) fit below 2 MHz, Cp alone above (Sec. 3.2.7).
  4. Computes water as a 1 cm thick layer (Cp alone, as in the paper).
  5. Writes one CSV file per run into results/ (same columns as extract_permittivity.py) and
     prints a table comparing the computed values with those printed in the paper.

The uncertainties quoted in the paper are computed by uncertainty.py, and the imaginary parts
by kramers_kronig.py.
"""

import csv
import os

import numpy as np

import extract_permittivity as ep
from ide_model import IDEModel, PAPER_MAIN_BATCH, PAPER_SECOND_BATCH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
RESULTS = os.path.join(HERE, 'results')

# Values printed in the paper: (description, computed-value key, printed value, number of decimals)
PAPER_VALUES = [
    ('quartz, main batch, 1.02 kHz (Sec. 4.1)', 'quartz_main_1k', 4.58, 2),
    ('quartz, main batch, 9.67 MHz', 'quartz_main_9M', 4.52, 2),
    ('quartz, second batch, 9.67 MHz', 'quartz_second_9M', 4.54, 2),
    ('water, 1.06 MHz', 'water_1M', 79.8, 1),
    ('MAPbI3 film s1, 9.67 MHz (Sec. 5.1, Fig. 7)', 'MAPI s1', 46.3, 1),
    ('MAPbI3 film s3, 9.67 MHz', 'MAPI s3', 45.9, 1),
    ('MAPbI3 film s4, 9.67 MHz', 'MAPI s4', 49.8, 1),
    ('PEA2PbI4 film s8, 9.67 MHz (Fig. 7)', 'PEPI s8', 9.9, 1),
    ('MAPbI3 (mean of 3 films), 9.67 MHz (Sec. 4.2)', 'MAPbI3', 47.3, 1),
    ('PEA2PbI4 (mean of 4 films), 9.67 MHz', 'PEA2PbI4', 10.5, 1),
    ('BA2PbI4, 9.67 MHz', 'BA2PbI4', 10.6, 1),
    ('HA2PbI4, 9.67 MHz', 'HA2PbI4', 9.3, 1),
    ('PEA2PbBr4 (run 3), 9.67 MHz', 'PEA2PbBr4', 6.1, 1),
    ('PEA2PbCl4 (run 3), 9.67 MHz', 'PEA2PbCl4', 6.0, 1),
    ('PEA2PbBr4 run 1, 9.67 MHz', 'PEPBr run 1', 6.00, 2),
    ('PEA2PbCl4 run 2, 9.67 MHz', 'PEPCl run 2', 6.12, 2),
    ('MAPbI3 (mean) near 15 kHz (Sec. 5.1: "about 58")', 'MAPbI3_15k', 58, 0),
    ('MAPbI3 (mean) at 100 kHz (Sec. 5.1: "51")', 'MAPbI3_100k', 51, 0),
]

# Compound values: mean over films (Sec. 4.2).  For PEA2PbBr4 and PEA2PbCl4 the paper uses run 3
# of the single film of each, and gives the other run as a check.
COMPOUND_RUNS = {
    'MAPbI3': ['MAPI s1', 'MAPI s3', 'MAPI s4'],
    'PEA2PbI4': ['PEPI s6', 'PEPI s7', 'PEPI s8', 'PEPI t5'],
    'BA2PbI4': ['BAPI c4'],
    'HA2PbI4': ['HAPI c7'],
    'PEA2PbBr4': ['PEPBr run 3'],
    'PEA2PbCl4': ['PEPCl run 3'],
}


def read_samples():
    with open(os.path.join(DATA, 'samples.csv'), newline='') as fh:
        lines = [l for l in fh if not l.startswith('#')]
    return list(csv.DictReader(lines))


def nearest(f, target):
    return int(np.argmin(np.abs(np.log(f / target))))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    samples = read_samples()
    geoms = {'main': PAPER_MAIN_BATCH, 'second': PAPER_SECOND_BATCH}
    computed, spectra = {}, {}

    # ---- quartz (blank substrates of the two wafer batches)
    for batch, blank in [('main', 'blank_main_batch.csv'), ('second', 'blank_second_batch.csv')]:
        res = ep.extract(os.path.join(DATA, blank), geom=geoms[batch])
        ep.write_csv(os.path.join(RESULTS, f'quartz_{batch}_batch.csv'), res, [f'blank file: {blank}'])
        f = res['f_Hz']
        if batch == 'main':
            computed['quartz_main_1k'] = res['eps_substrate'][nearest(f, 1.02e3)]
            computed['quartz_main_9M'] = res['eps_substrate'][nearest(f, 9.67e6)]
        else:
            computed['quartz_second_9M'] = res['eps_substrate'][nearest(f, 9.67e6)]
        print(f'quartz ({batch} batch) done', flush=True)

    # ---- films and water
    for smp in samples:
        label = smp['label']
        bulk = smp['thickness_nm'] == 'bulk'
        res = ep.extract(os.path.join(DATA, smp['blank_file']), os.path.join(DATA, smp['file']),
                         None if bulk else float(smp['thickness_nm']), geom=geoms[smp['batch']],
                         loss_cutoff_hz=0.0 if bulk else 2e6, bulk=bulk)
        ep.write_csv(os.path.join(RESULTS, label.replace(' ', '_') + '.csv'), res,
                     [f'sample: {label} ({smp["compound"]})', f'film file: {smp["file"]}',
                      f'blank file: {smp["blank_file"]}'])
        spectra[label] = res
        f = res['f_Hz']
        if bulk:
            computed['water_1M'] = res['eps_r'][nearest(f, 1.06e6)]
        else:
            computed[label] = res['eps_r'][nearest(f, 9.67e6)]
        print(f'{label} done', flush=True)

    # ---- compound means
    for comp, runs in COMPOUND_RUNS.items():
        computed[comp] = np.mean([computed[r] for r in runs])
    f = spectra['MAPI s1']['f_Hz']
    mapi = np.mean([spectra[r]['eps_r'] for r in COMPOUND_RUNS['MAPbI3']], axis=0)
    computed['MAPbI3_15k'] = mapi[nearest(f, 1.5e4)]
    computed['MAPbI3_100k'] = mapi[nearest(f, 1.0e5)]

    # ---- comparison table
    lines = [f'{"quantity":52s} {"computed":>10s} {"paper":>8s}  agrees to printed precision?']
    n_ok = 0
    for desc, key, printed, dec in PAPER_VALUES:
        val = computed[key]
        ok = abs(val - printed) <= (0.5 * 10 ** (-dec) + 1e-9 if dec > 0 else 1.0)
        n_ok += ok
        lines.append(f'{desc:52s} {val:10.4f} {printed:8.{dec}f}  {"yes" if ok else "NO"}')
    lines.append(f'{n_ok} of {len(PAPER_VALUES)} values agree.')
    lines.append('Compound means at 1.06 MHz (loss-corrected eps_r): ' + ', '.join(
        f'{c} {np.mean([spectra[r]["eps_r"][nearest(f, 1.06e6)] for r in runs]):.2f}' for c, runs in COMPOUND_RUNS.items()))
    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(RESULTS, 'summary.txt'), 'w') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
