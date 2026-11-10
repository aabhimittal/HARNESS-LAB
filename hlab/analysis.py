"""Load run traces and estimate factor effects.

Success model: logit P(success) = b0 + sum_j b_j x_j + sum_chains b_c (x_k x_l) + task_t, fitted as a binomial
GLM with a fixed effect per task. statsmodels has no frequentist binomial mixed model (only variational Bayes,
which understates uncertainty). In this balanced crossed design every task sees every config, so the task
fixed effect targets the same within-task effect as the spec's random intercept, with ordinary Wald CIs.
Tasks whose runs all pass or all fail carry no within-task information and are dropped from the fit.
Only one term per 2FI alias chain enters the model; its label lists the aliases it stands for.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit

from hlab.design import FACTORS, alias_chains

# $ per million tokens (input, output). shortcut: cache-token pricing ignored; add it once caching is used.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}


def load(trace_dir):
    """One row per finished run: config levels, task, outcome and costs."""
    rows = []
    for path in sorted(Path(trace_dir).glob("*.jsonl")):
        lines = path.read_text().splitlines()
        if not lines:
            continue
        start, end = json.loads(lines[0]), json.loads(lines[-1])
        if start.get("event") != "run_start" or end.get("event") != "run_end":
            continue
        usage = end.get("usage") or {}
        tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        price = PRICES.get(start["model"])
        rows.append({**{f: start["config"].get(f, -1) for f in FACTORS}, "task": start["task"],
                     "category": start.get("category") or "none", "seed": start["seed"], "model": start["model"],
                     "success": int(end["success"]), "input_tokens": tin, "output_tokens": tout,
                     "tokens": tin + tout, "cost": (tin * price[0] + tout * price[1]) / 1e6 if price else np.nan,
                     "wall_s": end["wall_s"], "steps": end.get("steps"), "trace": str(path)})
    return pd.DataFrame(rows)


def model_terms(df):
    """Estimable terms for the configs actually run: each main effect plus one representative per 2FI chain."""
    design = df[list(FACTORS)].drop_duplicates().to_dict("records")
    return [(chain[0], chain) for chain in alias_chains(design)]


def _matrix(df, terms, tasks):
    cols = {t: np.prod([df[f].to_numpy() for f in t], axis=0) for t, _ in terms}
    X = pd.DataFrame(cols, index=df.index, dtype=float)
    dummies = pd.get_dummies(df["task"]).reindex(columns=tasks).astype(float)
    return pd.concat([X, dummies], axis=1)  # task dummies replace the intercept


def fit_success(df):
    """Effect table: one row per estimable term, with log-odds coefficient (+/-1 coding) and pp effect at the
    overall success rate. A pp effect is the high-vs-low difference for a task of average difficulty."""
    rates = df.groupby("task")["success"].mean()
    used = df[df["task"].isin(rates[(rates > 0) & (rates < 1)].index)]
    terms = model_terms(df)
    if used.empty or used["task"].nunique() == 0:
        return pd.DataFrame(), {"tasks_used": 0, "tasks_dropped": int(rates.size), "runs": len(df)}
    tasks = sorted(used["task"].unique())
    res = sm.GLM(used["success"].to_numpy(), _matrix(used, terms, tasks), family=sm.families.Binomial()).fit()
    ci = res.conf_int()
    base = logit(np.clip(df["success"].mean(), 0.01, 0.99))
    pp = lambda b: expit(base + b) - expit(base - b)
    rows = []
    for term, chain in terms:
        b, lo, hi = res.params[term], ci.loc[term, 0], ci.loc[term, 1]
        rows.append({"term": term, "aliases": " = ".join(chain), "kind": "main" if len(term) == 1 else "2FI",
                     "coef": b, "se": res.bse[term], "ci_lo": lo, "ci_hi": hi, "z": b / res.bse[term],
                     "p": res.pvalues[term], "odds_ratio": np.exp(2 * b),
                     "pp": pp(b), "pp_lo": pp(lo), "pp_hi": pp(hi)})
    info = {"tasks_used": len(tasks), "tasks_dropped": int(rates.size - len(tasks)), "runs": len(used)}
    return pd.DataFrame(rows), info


def fit_secondary(df, metric):
    """% change in a cost metric for each factor's high vs low level (OLS on log1p, task fixed effects)."""
    d = df.dropna(subset=[metric])
    if d.empty or d[metric].nunique() < 2:
        return pd.DataFrame()
    terms = [(f, [f]) for f in FACTORS]
    tasks = sorted(d["task"].unique())
    res = sm.OLS(np.log1p(d[metric].to_numpy(dtype=float)), _matrix(d, terms, tasks)).fit()
    ci = res.conf_int()
    pct = lambda b: np.expm1(2 * b) * 100
    return pd.DataFrame([{"factor": f, "metric": metric, "pct": pct(res.params[f]), "pct_lo": pct(ci.loc[f, 0]),
                          "pct_hi": pct(ci.loc[f, 1])} for f in FACTORS])


def by_category(df, factor):
    """Success rate at each level of one factor, per task category (the spec asks for per-category results)."""
    t = df.pivot_table(index="category", columns=factor, values="success", aggfunc="mean")
    t = t.rename(columns={-1: "low", 1: "high"})
    t["diff"] = t.get("high") - t.get("low")
    return t.reset_index()


def examples(df, factor, n=2):
    """Tasks where the factor helped most and hurt most, with one trace from each level to read side by side."""
    g = df.groupby(["task", factor])["success"].mean().unstack()
    if g.shape[1] < 2:
        return [], []
    delta = (g[1] - g[-1]).sort_values()

    def pick(task, level, success):
        rows = df[(df["task"] == task) & (df[factor] == level) & (df["success"] == success)]
        return rows["trace"].iloc[0] if len(rows) else None

    out = lambda tasks, s: [{"task": t, "delta": delta[t], "trace_high": pick(t, 1, s), "trace_low": pick(t, -1, 1 - s)}
                            for t in tasks]
    return out(delta[delta > 0].index[::-1][:n], 1), out(delta[delta < 0].index[:n], 0)
