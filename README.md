# dielectric-IDEs

Python code to calculate the dielectric function of a thin film from capacitance measurements on
interdigitated electrodes (IDEs), using the model of

> C. E. McClure, K. R. Hansen, C. M. Shirley, C. Lindsay, L. Whittaker-Brooks, and J. S. Colton,
> "Measuring Dielectric Functions of Thin Films Using Interdigitated Electrodes, with Application
> to Metal Halide Perovskites" (2026).

Repository: https://github.com/coltonlab/dielectric-IDEs

The model treats the electrode thickness, the finite number of fingers, and the fields at the
finger tips and bus bars. It chooses the in-plane potential that minimizes the stored energy,
which gives an upper bound on the capacitance (Sec. 3.2 of the paper). The model works with
real or complex dielectric constants.

## Requirements

Python 3 with `numpy` and `scipy`; `matplotlib` only for the optional plots.

```
pip install -r requirements.txt
```

Tested with Python 3.12, numpy 2.4 and scipy 1.17.

## Quick start

1. Measure the capacitance Cp and conductance G versus frequency of a **blank** IDE (bare substrate)
   and of an IDE of the same design **with your film** on it.
2. Measure the film thickness and the electrode dimensions.
3. Run, for example with the data of the paper's MAPbI3 film s1:

```
python extract_permittivity.py --blank data/blank_main_batch.csv \
    --film data/MAPbI3_s1.csv --thickness-nm 437.8 --out MAPI_s1.csv --plot
```

This writes `MAPI_s1.csv`, which holds the substrate and film dielectric constants at every frequency,
and `MAPI_s1.png`, a quick plot. It also prints a summary at four frequencies; at 9.67 MHz it shows `film eps_r = 46.25`
(46.3 in Sec. 5.1 of the paper). Each film takes about 20 seconds.

Then the uncertainty and the imaginary part:

```
python uncertainty.py --blank data/blank_main_batch.csv \
    --film data/MAPbI3_s1.csv --thickness-nm 437.8 --out MAPI_s1_unc.csv
python kramers_kronig.py --input MAPI_s1.csv --plot
```

For your own electrodes, add `--s-um`, `--w-um`, `--te-um`, `--N`, `--L-mm` (gap, finger width and
electrode thickness in micrometers, number of fingers, overlap length in millimeters). The top of
`extract_permittivity.py` has more examples (substrate only; a liquid such as water) and describes
every output column.

## Data file format

Plain-text CSV, one row per frequency, no header required. Default columns, as written by our
HP 4192A measurement program in parallel mode:

| column | quantity | unit |
|---|---|---|
| 0 | frequency | Hz |
| 1 | admittance magnitude \|Y\| (not used) | S |
| 2 | parallel conductance G | S |
| 3 | parallel capacitance Cp | F |

For other layouts use `--columns f,G,Cp` with the column numbers counted from 0. Use -1 for G if your
files have no conductance; the loss correction is then skipped.

## Files

| file | what it is for |
|---|---|
| `extract_permittivity.py` | **Start here.** Blank + film data files in, dielectric function out (CSV and optional plot). |
| `ide_model.py` | The model itself: geometry, capacitance for given dielectric constants, and solving for the dielectric constants from measured capacitances. Read its header for the physics and the equations. |
| `uncertainty.py` | Uncertainty of eps_r, item by item, at every frequency (Sec. 4.2). Same inputs as `extract_permittivity.py`; the defaults are the values used in the paper, and each item can be changed. `--paper` reproduces the uncertainties of all films and compounds (1-2 minutes). |
| `kramers_kronig.py` | Imaginary part eps_i from eps_r by a Kramers-Kronig constrained fit, with the paper's test of where eps_i is determined by the data (Appendix B). Input: the CSV written by `extract_permittivity.py`. `--paper` analyzes all films, using data down to 5 Hz for the three thickest films as in the paper (about 7 minutes). |
| `fem2d.py` | Two-dimensional finite-element calculation of one period of the IDE, to check the model for a given geometry and film (Sec. 3.2.6, Appendix C). `--paper` reproduces the comparisons of Supplemental sections S4 and S9 (about 3 minutes). |
| `fem3d_end_correction.py` | Advanced. Three-dimensional finite-element calculation of the capacitance of the finger tips and bus bars, which led to the N_eff rule (Appendix C). Use it to check that rule for electrodes very different from ours. A run takes 10-20 seconds; `--paper` does all film types of the paper. |
| `make_si_data_files.py` | Writes the data files of Supplemental section S6 (eps_r, eps_i, uncertainty and the 'determined' flags for every film) from the outputs of `reproduce_paper.py` and `kramers_kronig.py --paper` (about 2 minutes). |
| `reproduce_paper.py` | Recomputes every dielectric constant reported in the paper from the raw data and compares with the printed values (about 4 minutes). |
| `check_model.py` | Quick self-test: comparison with the exact conformal-mapping result, numerical convergence, round trip. |
| `data/` | Raw data of the paper: `blank_main_batch.csv` and `blank_second_batch.csv` (bare IDEs of the two wafer batches), one file per film run named `<compound>_<sample>.csv` (13 runs on 11 films; the PEA2PbBr4 and PEA2PbCl4 films were each measured twice, `_run1`/`_run3` and `_run2`/`_run3`), and `water_1cm.csv`. `samples.csv` gives each file's sample label, compound, film thickness, wafer batch and blank, and the file's original name in our lab records. |
| `docs/` | Quick-start guide (Word and PDF) with worked examples for each script. |
| `results/` | Reference output: one CSV per run and `summary.txt` from `reproduce_paper.py`, the `*_KK.csv` files from `kramers_kronig.py --paper`, and `SI_data/`, the Supplemental data files. |

All scripts print their options with `--help`, and each begins with a description of what it does and
examples of how to run it.

## Notes

* **Frequency range.** By default the analysis runs from 100 Hz to 10 MHz (`--fmin`, `--fmax`). Below
  about 100 Hz the capacitance of our bare substrates rises steeply, an instrumental effect. Toward
  13 MHz the analyzer's errors grow (Secs. 2 and 4.2 of the paper).
* **Loss correction.** Below 2 MHz (`--loss-cutoff`) the film is fitted with complex dielectric
  constants to the measured Cp - iG/omega. Fitting Cp alone overestimates eps' of a lossy film. Above
  2 MHz our measured G had a setup-dependent error, so there the code uses Cp alone. Check where your
  own G data are reliable. The eps'' obtained from G includes DC conduction, and near 1 MHz it can be
  negative because of that error. The paper's eps_i comes from a Kramers-Kronig analysis of eps_r.
* **N_eff.** The code uses the paper's rule N_eff = N - 1/2 + N a/L (a = s + w).
* **Supplemental data files.** The data files of Supplemental section S6 (`S6_data_*.csv`) were made with
  this code: run `reproduce_paper.py`, then `kramers_kronig.py --paper`, then `make_si_data_files.py`, which
  writes them to `results/SI_data/`. The copies in `results/SI_data/` are identical to the published files.
* **Film thickness** dominates the uncertainty of eps_r (Sec. 4.2), so measure it carefully.

## How to cite

If you use this code or data, please cite the paper above and this repository
(https://github.com/coltonlab/dielectric-IDEs).

## License

MIT License; see `LICENSE`.
