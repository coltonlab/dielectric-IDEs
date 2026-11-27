"""
table_s1.py -- Supplemental Table S1 of the paper: the time to extract the dielectric constants at the
78 frequencies on one processing core, for our model and for each numerical method on the cheapest mesh
at least as accurate as our model, with the ratio to our model's time.

    py table_s1.py                 (reads this folder's results/; instant)

Inputs (all measured on our desktop PC, Intel Core Ultra 7 265, one core):
  results/core_extraction.txt   (core_extraction_timing.py)
      section 1: the numbers of real and complex evaluations of the capacitance in the extraction of all
                 our data (2 blanks + 13 film measurements, 9 combinations of electrodes and film
                 thickness), the wall times of our model for all data, and our model's time per real and
                 per complex evaluation for the MAPbI3 films (t = 438 nm);
      section 2: setup time and the time of one real and one complex solve of each direct method on the
                 test problem.
  results/benchmark_results.csv (bench_sor_graded.py, bench_sor_squares.py)
      the newest run of the first relaxation (SOR) rung that reaches our model's accuracy, started from
      V = 0, for the graded and the uniform grid.

"One film" is one MAPbI3 film with the blank already known: the evaluations of the three MAPbI3 films
divided by three.  "All data" is our 13 film measurements and the two blanks.  "Cp only" uses the
capacitance alone at every frequency; "complex fit" also fits Cp - iG/omega below 2 MHz, as for the values
in the paper.  For the numerical methods: one mesh per geometry plus one solve per evaluation.  Relaxation
gets a range for the complex fit, a complex solve taken to cost 1-2 times a real one; it gets no setup time.
The paper rounds these numbers.
Written with Claude, Nov 2026.
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

UNIT = {'us': 1e-6, 'ms': 1e-3, 's': 1.0, 'min': 60.0, 'h': 3600.0}


def seconds(value, unit):
    return float(value.replace(',', '')) * UNIT[unit]


def count(text):
    return int(text.replace(',', ''))


def read_core_extraction(path):
    txt = open(path, encoding='utf-8').read()
    out = {}
    m = re.search(r'paper\s*:\s*([\d,]+) real \+ ([\d,]+) complex.*?(\d+) geometries.*?wall ([\d.]+) (us|ms|s)', txt)
    out['real_all'], out['cx_all'], out['geoms'] = count(m.group(1)), count(m.group(2)), int(m.group(3))
    out['wall_complex_fit'] = seconds(m.group(4), m.group(5))
    m = re.search(r'Cp only\s*:\s*([\d,]+) real \+ 0 complex.*?wall ([\d.]+) (us|ms|s)', txt)
    assert count(m.group(1)) == out['real_all']
    out['wall_cp_only'] = seconds(m.group(2), m.group(3))
    m = re.search(r's = 50\.27 um, t = 438 nm\s+real\s+([\d,]+) x\s+([\d.]+) (us|ms)\s+complex\s+([\d,]+) x\s+([\d.]+) (us|ms)', txt)
    out['real_mapi3'], out['t_real_mapi'] = count(m.group(1)), seconds(m.group(2), m.group(3))
    out['cx_mapi3'], out['t_cx_mapi'] = count(m.group(4)), seconds(m.group(5), m.group(6))
    sec2 = txt[txt.index('2. ONE CAPACITANCE'):txt.index('3. ESTIMATED')]
    methods = []
    for line in sec2.splitlines():
        m = re.match(r'\s+(.+?)\s{2,}([\d,]+)\s+([\d.]+) (us|ms|s)\s+([\d.]+) (us|ms|s)\s+([\d.]+) (us|ms|s)\s', line)
        if m:
            methods.append(dict(name=m.group(1), nodes=count(m.group(2)), setup=seconds(m.group(3), m.group(4)),
                                real=seconds(m.group(5), m.group(6)), cx=seconds(m.group(7), m.group(8))))
    out['methods'] = methods
    return out


def newest_first_hit(path, method):
    """total_s and n_nodes of the newest run of the first rung of `method` that reached the model's accuracy."""
    rows = [r for r in csv.DictReader(open(path, newline='', encoding='utf-8'))
            if r['method'] == method and not r['label'].startswith('check')]
    newest = max(r['timestamp'] for r in rows)
    day = newest[:10]                                  # the rungs of the newest run (same day)
    run = sorted((r for r in rows if r['timestamp'][:10] == day), key=lambda r: int(r['n_nodes']))
    hit = next(r for r in run if r['beats_model'] == '1')
    return float(hit['total_s']), int(hit['n_nodes'])


def num(x):
    """Three significant digits, never in exponent form."""
    s = f'{x:.3g}'
    return f'{x:.0f}' if 'e' in s else s


def fmt(t):
    if t < 1:
        return f'{num(t * 1e3)} ms'
    if t < 120:
        return f'{num(t)} s'
    if t < 7200:
        return f'{num(t / 60)} min'
    if t < 3 * 86400:
        return f'{num(t / 3600)} h'
    return f'{num(t / 86400)} days'


def main():
    ce = read_core_extraction(os.path.join(RES, 'core_extraction.txt'))
    r1, c1 = ce['real_mapi3'] / 3, ce['cx_mapi3'] / 3              # one MAPbI3 film
    rall, call, g = ce['real_all'], ce['cx_all'], ce['geoms']
    model = ce['methods'][0]
    assert model['name'] == 'Our model'
    # our model: one film from its time per evaluation at 438 nm; all data measured (wall time)
    m1 = model['setup'] + r1 * ce['t_real_mapi']
    base = [m1, m1 + c1 * ce['t_cx_mapi'], ce['wall_cp_only'], ce['wall_complex_fit']]
    head = ['Method', 'Nodes', 'One film: Cp only', 'One film: complex fit', 'All data: Cp only',
            'All data: complex fit']
    rows = [['Our model', f"{model['nodes']:,}"] + [fmt(t) for t in base]]
    for mt in ce['methods'][1:]:
        a = mt['setup'] + r1 * mt['real']
        b = a + c1 * mt['cx']
        c = g * mt['setup'] + rall * mt['real']
        d = c + call * mt['cx']
        name = mt['name'].replace('Appendix C meshes', 'meshes of Section S4')   # older label in the log
        rows.append([name, f"{mt['nodes']:,}"] + [f'{fmt(t)} ({t / b0:,.0f})' for t, b0 in zip((a, b, c, d), base)])
    csvp = os.path.join(RES, 'benchmark_results.csv')
    for name, method in (('Finite differences, graded grid, relaxation (SOR)', 'five-point, SOR (graded grid)'),
                         ('Finite differences, uniform grid, relaxation (SOR)', 'SOR, uniform squares')):
        t, n = newest_first_hit(csvp, method)
        a = r1 * t
        b = (a + c1 * t, a + 2 * c1 * t)
        c = rall * t
        d = (c + call * t, c + 2 * call * t)
        rows.append([name, f'{n:,}', f'{fmt(a)} ({a / base[0]:,.0f})',
                     f'{fmt(b[0])}-{fmt(b[1])} ({b[0] / base[1]:,.0f}-{b[1] / base[1]:,.0f})',
                     f'{fmt(c)} ({c / base[2]:,.0f})',
                     f'{fmt(d[0])}-{fmt(d[1])} ({d[0] / base[3]:,.0f}-{d[1] / base[3]:,.0f})'])
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print('  '.join(x.ljust(w) for x, w in zip(r, widths)).rstrip())
    print(f'\nEvaluations: one MAPbI3 film {r1:,.0f} real + {c1:,.0f} complex; all data {rall:,} real + {call:,} '
          f'complex, {g} geometries.  Ratios (in parentheses) are to our model in the same column.')


if __name__ == '__main__':
    main()
