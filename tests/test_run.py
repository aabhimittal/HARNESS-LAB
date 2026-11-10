import json

import pytest

from hlab.run import load_tasks, run_one
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


def test_model_error_is_a_failed_run_not_a_crash(tmp_path, monkeypatch):
    import hlab.run

    class Boom:
        def step(self, *a):
            raise RuntimeError("api down")

    monkeypatch.setattr(hlab.run, "make_model", lambda name, task: Boom())
    r = run_one(TASKS[0], "x", 0, tmp_path)
    assert not r["success"] and r["outcome"].startswith("error: RuntimeError")
    assert _events(r["trace"])[-1]["event"] == "run_end"
