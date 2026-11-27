"""
bench_model.py -- (a) the paper's model (ide_model.py): time and accuracy for the test problem.

What is timed
-------------
  setup   IDEModel(geom): the nodes of f(x), the matrix of the sum over ALL Fourier modes (closed
          form) and the rows of the first 1024 modes.  Done once per electrode geometry.
  S       one evaluation of S for given dielectric constants: a sum over the modes that feel the
          top of the film (the first 772 here) and a linear solve of about 40 equations.  This is
          what is repeated thousands of times when a spectrum is analyzed, so it is also reported
          as "each further S" (per_eval_s).
  eps     model.solve_film(C): the actual use, finding eps_film from a capacitance (root finding,
          about 10 evaluations of S).  Printed, not logged.

Rungs
-----
  legacy   the Fourier sums cut off after 20,000 terms, as in the code before Nov. 2026 (the setting
           behind the paper's numbers): same S as the old code to 1e-13, but timed with the new code,
           which splits the sums the same way.  (The old code itself took about 55 ms to set up and
           4 ms per S in the sandbox, because it summed all 20,000 terms for every S.)
  default  default settings, exact sums: the model's accuracy target, bench_common.S_MODEL.
  finer in-plane nodes, exact sums.  These lower S only slightly: the model's error has a floor of
           about +0.0098% set by its physical approximation (the potential between the electrodes
           is taken independent of y), not by its numerical settings.
  check    the convergence check a user could make: the default with every node spacing halved
           (h_edge t_e/16, h_max 1 um, growth 0.075), logged as 'check: default (exact sums)'.  It
           sees only the discretization error (about 1e-5), not the model's error of +0.011%,
           which is known only from comparisons such as the FEM of the paper.

Run:   python bench_model.py          (about 5 s)

Expected (Claude's sandbox, Xeon 2.1 GHz shared with other jobs, 21 Nov 2026; your times will be
shorter).  Error relative to bench_common.S_REF:
    rung                                                nodes  S            error     setup    each S
    legacy: sums cut off at 20,000 terms (old default)     43  7.188918843  +0.01136%  ~47 ms   ~0.2 ms
    default (exact sums)                                   43  7.188918981  +0.01136%  ~2.5 ms  ~0.2 ms
    growth 0.075                                           69  7.188840886  +0.01027%  ~4 ms    ~0.4 ms
    growth 0.0375, h_max 0.5 um                           122  7.188821871  +0.01000%  ~8 ms    ~1.0 ms
    + h_edge t_e/16                                       141  7.188813986  +0.00990%  ~10 ms   ~1.3 ms
    growth 0.02, h_max 0.25, h_edge t_e/32                267  7.188808335  +0.00982%  ~29 ms   ~3.6 ms
    check: default (exact sums)                            82  7.188830453  +0.01013%  ~4.6 ms  ~0.5 ms
    The check estimates the default's error as +1.23e-05 (its true error is +1.136e-04).
    eps_film recovered from C in ~2 ms, with exact or cut-off sums (the old code took ~35 ms).
"""
import bench_common as bc          # first: pins numpy to one thread

from ide_model import IDEGeometry, IDEModel

DEFAULT = 'default (exact sums)'
RUNGS = [
    ('legacy: sums cut off at 20,000 terms (old default)', dict(n_terms=20000)),
    (DEFAULT, {}),
    ('growth 0.075', dict(growth=0.075)),
    ('growth 0.0375, h_max 0.5 um', dict(growth=0.0375, h_max_um=0.5)),
    ('+ h_edge t_e/16', dict(growth=0.0375, h_max_um=0.5, h_edge_um=bc.T_E / 16)),
    ('growth 0.02, h_max 0.25, h_edge t_e/32', dict(growth=0.02, h_max_um=0.25, h_edge_um=bc.T_E / 32)),
]
# The check: every node spacing of the default halved (default h_edge = t_e/8, h_max = 2 um, growth 0.15)
CHECK = dict(h_edge_um=bc.T_E / 16, h_max_um=1.0, growth=0.075)


def run(g, kw):
    """Median setup time, the model, S and the median time of one further S."""
    legacy = 'n_terms' in kw
    model, t_setup = bc.median_time(lambda: IDEModel(g, **kw), 5 if legacy else 21)
    S, t_S = bc.median_time(lambda: model.S(bc.EPS_FILM, bc.EPS_SUB, bc.T_P), 21 if legacy else 101)
    return model, S, t_setup, t_S


def describe(model):
    if model.n_terms is not None:
        return (f'{len(model.x)} in-plane nodes (n_nodes), sums cut off after {model.n_terms} Fourier terms '
                f'(same S as the code before Nov 2026; timed with the new code)')
    n1 = model._n_film_modes(bc.T_P - bc.T_E)
    return f'{len(model.x)} in-plane nodes (n_nodes), exact sums ({n1} modes feel the film top; the rest in closed form)'


def main():
    bc.banner("(a) ide_model.py -- the paper's model")
    g = IDEGeometry(s_um=bc.S_GAP, w_um=bc.W_FINGER, te_um=bc.T_E)     # all dimensions explicit
    print(f"{'rung':52s} {'nodes':>5s} {'S':>13s} {'S/S_REF-1':>10s} {'setup ms':>9s} {'each S ms':>10s}")
    S_default = None
    for label, kw in RUNGS:
        model, S, t_setup, t_S = run(g, kw)
        print(f'{label:52s} {len(model.x):5d} {S:13.9f} {bc.rel_err(S):+10.3e} {1000 * t_setup:9.2f} {1000 * t_S:10.3f}',
              flush=True)
        bc.record('model (ide_model.py)', label, len(model.x), t_setup, t_S, S, per_eval_s=t_S,
                  notes=describe(model), quiet=True)
        if label == DEFAULT:
            S_default = S
            if abs(S - bc.S_MODEL) > 1e-9:
                print(f'NOTE: default S = {S!r} differs from bc.S_MODEL = {bc.S_MODEL}; update bench_common.py')

    # The convergence check a user could make without a reference value (see the docstring).
    model, S, t_setup, t_S = run(g, CHECK)
    label = 'check: ' + DEFAULT
    est = S_default / S - 1
    print(f'{label:52s} {len(model.x):5d} {S:13.9f} {bc.rel_err(S):+10.3e} {1000 * t_setup:9.2f} {1000 * t_S:10.3f}')
    print(f'   the check estimates the default\'s error as {est:+.2e}; its true error is {bc.rel_err(S_default):+.2e} '
          f'(the check sees only the discretization error)')
    bc.record('model (ide_model.py)', label, len(model.x), t_setup, t_S, S, per_eval_s=t_S,
              notes=(f'every node spacing / 2 (h_edge t_e/16, h_max 1 um, growth 0.075); estimate S_base/S_check - 1 = '
                     f'{est:+.3e}; true error of base {bc.rel_err(S_default):+.3e}; sees only discretization error, '
                     f'not the model error'), quiet=True)

    # The real use: recover eps_film from a capacitance (here the model's own value for eps 47).
    for name, kw in [('exact sums', {}), ('legacy, 20,000 terms', dict(n_terms=20000))]:
        model = IDEModel(g, **kw)
        C = g.prefactor() * model.S(bc.EPS_FILM, bc.EPS_SUB, bc.T_P)
        eps, t = bc.median_time(lambda: model.solve_film(C, bc.EPS_SUB, bc.T_P), 21 if not kw else 5)
        print(f'\nExtracting eps_film from a capacitance ({name}): {eps:.6f} in {1000 * t:.2f} ms '
              f'(setup not included)', end='')
    print()


if __name__ == '__main__':
    main()
