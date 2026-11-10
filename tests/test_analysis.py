"""Plant known effects in synthetic traces and check the analysis and effect cards recover them."""
import itertools
import json

import numpy as np
import pytest
from scipy.special import expit

from hlab import analysis, report
from hlab.design import fractional_factorial
from hlab.middleware import config_name

TRUE = {"A": 0.5, "B": 0.0}  # +/-1-scale log-odds coefficients
TRUE_AB = 0.4  # planted A x B interaction (its chain is AB = CF = EG)


def _write_traces(directory, n_tasks=30, n_seeds=4, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, n_tasks)
    for t, cfg, s in itertools.product(range(n_tasks), fractional_factorial(), range(n_seeds)):
        eta = u[t] + TRUE["A"] * cfg["A"] + TRUE["B"] * cfg["B"] + TRUE_AB * cfg["A"] * cfg["B"]
        tokens = int(1000 * (1.5 if cfg["F"] > 0 else 1.0) * rng.lognormal(0, 0.1))
        run_id = f"{config_name(cfg)}__t{t}__s{s}"
        start = {"event": "run_start", "run_id": run_id, "config": cfg, "task": f"t{t}",
                 "category": "even" if t % 2 == 0 else "odd", "seed": s, "model": "claude-haiku-5-5"}
        end = {"event": "run_end", "run_id": run_id, "success": bool(rng.random() < expit(eta)), "wall_s": 1.0,
               "steps": 3, "usage": {"input_tokens": tokens, "output_tokens": tokens // 10}}
        (directory / f"{run_id}.jsonl").write_text(json.dumps(start) + "\n" + json.dumps(end) + "\n")


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    d = tmp_path_factory.mktemp("traces")
    _write_traces(d)
    (d / "unfinished.jsonl").write_text(json.dumps({"event": "run_start"}) + "\n")
    return analysis.load(d)


def test_load_skips_unfinished_runs_and_prices_tokens(df):
    assert len(df) == 30 * 16 * 4 and df["cost"].gt(0).all()


def test_planted_effects_are_recovered(df):
    effects, info = analysis.fit_success(df)
    e = effects.set_index("term")
    assert e.loc["A", "ci_lo"] < TRUE["A"] < e.loc["A", "ci_hi"] and e.loc["A", "ci_lo"] > 0
    assert e.loc["B", "ci_lo"] < 0 < e.loc["B", "ci_hi"]
    assert e.loc["AB", "aliases"] == "AB = CF = EG" and e.loc["AB", "ci_lo"] < TRUE_AB < e.loc["AB", "ci_hi"]
    assert e.loc["AB", "ci_lo"] > 0 and len(effects) == 7 + 7 and info["tasks_used"] <= 30


def test_secondary_cost_effect(df):
    sec = analysis.fit_secondary(df, "tokens").set_index("factor")
    assert sec.loc["F", "pct_lo"] < 50 < sec.loc["F", "pct_hi"] and sec.loc["A", "pct_lo"] < 0 < sec.loc["A", "pct_hi"]


def test_constant_tasks_are_dropped(df):
    d = df.copy()
    d.loc[d["task"] == "t0", "success"] = 1
    _, before = analysis.fit_success(df)
    _, after = analysis.fit_success(d)
    assert after["tasks_dropped"] == before["tasks_dropped"] + 1 and after["runs"] == before["runs"] - 64


def test_effect_cards(df, tmp_path):
    index = report.write(df, tmp_path)
    html = index.read_text()
    assert "AB = CF = EG" in html and 'href="factor_A.html"' in html
    card = (tmp_path / "factor_A.html").read_text()
    assert "Main effect" in card and "Where it helped" in card and "even" in card and ".jsonl" in card
    assert len(list(tmp_path.glob("factor_*.html"))) == 7 and (tmp_path / "effects.csv").exists()


def test_report_without_variation(df, tmp_path):
    d = df.assign(success=1)
    assert "nothing to estimate" in report.write(d, tmp_path).read_text()
