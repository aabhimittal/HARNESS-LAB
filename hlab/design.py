"""Experiment design: 2^(7-3) resolution IV fractional factorial, alias structure, fold-over, power.

    python -m hlab.design                          # the 16 configs and their alias chains
    python -m hlab.design power --tasks 40 --seeds 3
"""
import argparse
import itertools
import math

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

FACTORS = "ABCDEFG"
# Res IV generators, chosen (by search over relabelings) so that DE, AB, AE, CE, AC and EF each sit in a
# different alias chain: those are the interactions most likely to be real (loops x verification, the two
# context strategies, plan x restarts). The textbook E=ABC would alias AB with CE.
GENERATORS = {"E": "BCD", "F": "ABC", "G": "ACD"}


def fractional_factorial(base="ABCD", generators=GENERATORS):
    """Configs in standard order, each a {factor: -1|+1} dict."""
    design = []
    for levels in itertools.product([-1, 1], repeat=len(base)):
        row = dict(zip(base, levels))
        row.update({g: math.prod(row[c] for c in word) for g, word in generators.items()})
        design.append(dict(sorted(row.items())))
    return design


def full_factorial(factors=FACTORS):
    return [dict(zip(factors, levels)) for levels in itertools.product([-1, 1], repeat=len(factors))]


def foldover(design, factor):
    """Append the design with one factor's sign flipped: frees that factor's 2FIs from their alias chains."""
    return design + [{**r, factor: -r[factor]} for r in design]


def _column(design, term):
    return np.array([math.prod(r[f] for f in term) for r in design])


def alias_chains(design, order=2):
    """Group main effects and 2FIs whose columns are equal up to sign. The first term of each chain is
    the one a model can estimate; the rest are its aliases. Columns that are constant are dropped."""
    factors = sorted(design[0])
    terms = list(factors) + ["".join(p) for p in itertools.combinations(factors, 2)][: None if order >= 2 else 0]
    chains = {}
    for t in terms:
        col = _column(design, t)
        if abs(col.sum()) == len(col):
            continue
        key = tuple(col * col[0])
        chains.setdefault(key, []).append(t)
    return list(chains.values())


def power(n_tasks, n_seeds, n_configs=16, p0=0.5, sigma=1.0, tau=0.2, alpha=0.05, target=0.8):
    """Minimum detectable effect for any main effect or 2FI chain (all columns are orthogonal +/-1).

    Model: logit p = logit(p0) + u_task + beta_j x_j, u ~ N(0, sigma^2). The coefficient is on the +/-1 scale,
    so the high-vs-low log-odds difference is 2*beta. Information per run is p(1-p), averaged over tasks.
    tau is the SD of the factor's effect across tasks: that part of the error only shrinks with more tasks,
    which is why adding tasks beats adding seeds.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(60)
    p = expit(logit(p0) + sigma * nodes)
    info = (weights * p * (1 - p)).sum() / weights.sum()
    se = math.sqrt(1 / (n_tasks * n_seeds * n_configs * info) + tau**2 / n_tasks)
    beta = (norm.ppf(1 - alpha / 2) + norm.ppf(target)) * se
    return {"tasks": n_tasks, "seeds": n_seeds, "runs": n_tasks * n_seeds * n_configs, "se": se,
            "mde_logodds": 2 * beta, "mde_pp": expit(logit(p0) + beta) - expit(logit(p0) - beta)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    pw = sub.add_parser("power", help="minimum detectable effect")
    pw.add_argument("--tasks", type=int, nargs="+", default=[10, 20, 40, 80])
    pw.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 5])
    pw.add_argument("--p0", type=float, default=0.5, help="success rate of a typical task")
    pw.add_argument("--sigma", type=float, default=1.0, help="SD of task difficulty (logit scale)")
    pw.add_argument("--tau", type=float, default=0.2, help="SD of the factor effect across tasks")
    a = p.parse_args(argv)
    if a.cmd == "power":
        print(f"MDE in percentage points at p0={a.p0} (alpha 0.05, power 0.8), 16 configs")
        print("tasks\\seeds " + "".join(f"{s:>8}" for s in a.seeds))
        for t in a.tasks:
            row = [power(t, s, p0=a.p0, sigma=a.sigma, tau=a.tau)["mde_pp"] * 100 for s in a.seeds]
            print(f"{t:>11} " + "".join(f"{v:>7.1f}%" for v in row))
        return
    d = fractional_factorial()
    print("config  " + " ".join(FACTORS))
    for i, r in enumerate(d):
        print(f"{i:>6}  " + " ".join("+" if r[f] > 0 else "-" for f in FACTORS))
    print("\nalias chains (2FIs; main effects are clear of them):")
    for chain in alias_chains(d):
        if len(chain[0]) == 2:
            print("  " + " = ".join(chain))


if __name__ == "__main__":
    main()
