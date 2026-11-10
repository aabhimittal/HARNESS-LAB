import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from hlab.design import fractional_factorial
from hlab.report import main as report_main
from hlab.run import load_tasks, run_one
from hlab.run import main as run_main
from tests.test_sandbox import pytestmark  # noqa: F401  (skip without docker)

TASKS = load_tasks()


def _events(path):
    return [json.loads(line) for line in open(path)]


def test_smoke_suite_has_five_unique_tasks():
    assert len(TASKS) == 5 and len({t["id"] for t in TASKS}) == 5


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_oracle_passes_and_writes_trace(task, tmp_path):
    r = run_one(task, "oracle", 0, tmp_path)
    assert r["success"], r
    ev = _events(r["trace"])
    assert ev[0]["event"] == "run_start" and ev[-1]["event"] == "run_end"
    assert [e["event"] for e in ev if e["event"] in ("model", "tool")] == ["model", "tool", "model"]
    assert ev[-1]["success"] is True and ev[-1]["outcome"] == "end_turn"


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_null_model_fails(task, tmp_path):
    # Checks must not pass vacuously on an untouched container.
    assert not run_one(task, "null", 0, tmp_path)["success"]


CORE = load_tasks("core")


def _parallel(fn, items):
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(fn, items))


def test_core_suite_is_solvable_and_not_vacuous(tmp_path):
    assert len({t["id"] for t in TASKS + CORE}) == len(TASKS + CORE) == 20
    oracle = _parallel(lambda t: run_one(t, "oracle", 0, tmp_path / "o"), CORE)
    null = _parallel(lambda t: run_one(t, "null", 0, tmp_path / "n"), CORE)
    assert [r["run_id"] for r in oracle if not r["success"]] == []
    assert [r["run_id"] for r in null if r["success"]] == []


def test_every_design_config_runs_end_to_end_resumes_and_reports(tmp_path, capsys):
    """All 16 middleware stacks drive a real container; a rerun skips finished runs; the report builds."""
    task = next(t for t in CORE if t["id"] == "three-parts")
    results = _parallel(lambda c: run_one(task, "oracle", 0, tmp_path / "tr", c), fractional_factorial())
    assert all(r["success"] for r in results) and len({r["run_id"] for r in results}) == 16
    assert run_one(task, "oracle", 0, tmp_path / "tr", fractional_factorial()[5]).get("skipped")
    ralph = _events(next(r["trace"] for r in results if "E+" in r["run_id"]))
    assert any(e["event"] == "ralph_restart" for e in ralph)  # the loop really fires inside a real run
    run_main(["-j", "4", "--model", "null", "--tasks", "smoke", "--out", str(tmp_path / "tr")])
    report_main([str(tmp_path / "tr"), "--out", str(tmp_path / "rep")])
    assert (tmp_path / "rep" / "index.html").exists()


def test_every_task_declares_outputs():
    assert all(t.get("outputs") and all(p.startswith("/work/") for p in t["outputs"]) for t in TASKS + CORE)


TAMPER = {  # task id -> commands that fake a pass by rigging what an in-place check would trust
    "log-errors": ["echo '2026 INFO [api] x' > /work/app.log", "echo '{}' > /work/errors.json"],
    "write-tests": ["printf 'import unittest\\n' > /work/test_inventory.py",
                    "echo 'raise SystemExit' > /work/inventory.py"],
    "needle-output": ["printf 'print(\"id=31337 checksum=abc\")\\n' > /work/report.py", "echo abc > /work/checksum.txt"],
    "csv-join": ["printf 'region,total\\n' > /work/region_totals.csv",
                 "printf 'id,name,region\\n' > /work/customers.csv", "printf 'order_id,customer_id,amount\\n' > /work/orders.csv"],
    # root in its own container: swap the interpreter for one that always succeeds
    "two-bugs": ["printf '#!/bin/sh\\nexit 0\\n' > /usr/local/bin/python3.12"],
}


@pytest.mark.parametrize("task_id", TAMPER)
def test_grader_ignores_tampering_in_agent_container(task_id, tmp_path, monkeypatch):
    import hlab.run
    from hlab.core.agent import ScriptedModel

    monkeypatch.setattr(hlab.run, "make_model", lambda name, task: ScriptedModel(TAMPER[task_id]))
    task = next(t for t in CORE if t["id"] == task_id)
    r = run_one(task, "tamper", 0, tmp_path)
    assert not r["success"], f"tampering fooled the grader on {task_id}"


def test_model_error_is_a_failed_run_not_a_crash(tmp_path, monkeypatch):
    import hlab.run

    class Boom:
        def step(self, *a):
            raise RuntimeError("api down")

    monkeypatch.setattr(hlab.run, "make_model", lambda name, task: Boom())
    r = run_one(TASKS[0], "x", 0, tmp_path)
    assert not r["success"] and r["outcome"].startswith("error: RuntimeError")
    assert _events(r["trace"])[-1]["event"] == "run_end"
