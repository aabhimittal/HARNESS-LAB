import itertools

import numpy as np
import pandas as pd
from scipy.special import expit

from hlab.analysis import fit_success
from hlab.design import FACTORS, alias_chains, foldover, fractional_factorial, full_factorial, power


def _X(design):
    return np.array([[r[f] for f in FACTORS] for r in design])


def test_fractional_factorial_is_balanced_and_orthogonal():
    d = fractional_factorial()
    X = _X(d)
    assert X.shape == (16, 7) and len({tuple(r) for r in X}) == 16
    assert (X.T @ X == 16 * np.eye(7)).all()


def test_resolution_iv_and_chosen_interactions_are_separated():
    chains = alias_chains(fractional_factorial())
    assert all(len(c) == 1 for c in chains if len(c[0]) == 1)  # no main effect aliased with a 2FI
    assert sum(len(c) for c in chains if len(c[0]) == 2) == 21 and all(len(c) == 3 for c in chains if len(c[0]) == 2)
    chain_of = {t: i for i, c in enumerate(chains) for t in c}
    interest = ["DE", "AB", "AE", "CE", "AC", "EF"]
    assert len({chain_of[t] for t in interest}) == len(interest)


def test_single_factor_foldover_frees_that_factors_interactions():
    chains = alias_chains(foldover(fractional_factorial(), "D"))
    for c in chains:
        if any("D" in t and len(t) == 2 for t in c):
            assert len(c) == 1, c


def test_full_factorial_has_every_term_clear():
    chains = alias_chains(full_factorial())
    assert len(full_factorial()) == 128 and len(chains) == 7 + 21 and all(len(c) == 1 for c in chains)


def test_power_more_tasks_beat_more_seeds_at_equal_cost():
    assert power(80, 1)["mde_pp"] < power(40, 2)["mde_pp"] < power(20, 4)["mde_pp"]
    assert power(40, 3)["mde_pp"] < power(40, 2)["mde_pp"] < power(20, 2)["mde_pp"]
    assert power(40, 2, tau=0)["mde_pp"] == power(80, 1, tau=0)["mde_pp"]  # only effect heterogeneity separates them


def test_power_standard_error_matches_simulation():
    """Simulate the analysis model with no true effects and compare the spread of fitted coefficients to the
    analytic SE the power calculator uses."""
    rng = np.random.default_rng(0)
    T, S, sigma = 40, 2, 1.0
    design = fractional_factorial()
    coefs = []
    for _ in range(30):
        u = rng.normal(0, sigma, T)
        rows = [{**cfg, "task": f"t{t}", "seed": s, "success": int(rng.random() < expit(u[t]))}
                for t, cfg, s in itertools.product(range(T), design, range(S))]
        effects, _ = fit_success(pd.DataFrame(rows))
        coefs.append(effects.set_index("term").loc["A", "coef"])
    predicted = power(T, S, sigma=sigma, tau=0)["se"]
    assert 0.75 < np.std(coefs) / predicted < 1.3, (np.std(coefs), predicted)
