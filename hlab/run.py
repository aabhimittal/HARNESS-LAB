"""Experiment runner: configs x tasks x seeds, a fresh container per run, one JSONL trace per run.

    python -m hlab.run --model oracle                        # baseline config, smoke suite, no API key
    python -m hlab.run --design --tasks core --seeds 3 -j 8  # all 16 fractional-factorial configs
    python -m hlab.run --config A+B-C-D+E-F-G- --model claude-haiku-5-5
Runs whose trace already ends in run_end are skipped, so an interrupted sweep resumes where it stopped.
"""
import argparse
import json
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hlab.core.agent import AnthropicModel, ScriptedModel, run_agent
from hlab.design import FACTORS, fractional_factorial
from hlab.middleware import build, config_name
from hlab.sandbox import DEFAULT_IMAGE, Sandbox

TASKS_DIR = Path(__file__).parent / "tasks"
CHECK_TIMEOUT_S = 300
BASELINE = {f: -1 for f in FACTORS}


def load_tasks(suite="smoke"):
    path = TASKS_DIR / f"{suite}.toml" if "/" not in suite and not suite.endswith(".toml") else Path(suite)
    with open(path, "rb") as f:
        return tomllib.load(f)["task"]


def parse_config(name):
    """'A+B-C-D+E-F-G-' -> {'A': 1, 'B': -1, ...}"""
    assert len(name) == 14 and all(name[i] in FACTORS and name[i + 1] in "+-" for i in range(0, 14, 2)), name
    return {name[i]: 1 if name[i + 1] == "+" else -1 for i in range(0, 14, 2)}


def make_model(name, task):
    if name == "oracle":
        return ScriptedModel([task["solution"]], name="oracle")
    if name == "null":
        return ScriptedModel([], name="null")
    return AnthropicModel(name)


def _finished(path):
    try:
        return json.loads(path.read_text().splitlines()[-1])["event"] == "run_end"
    except (OSError, IndexError, ValueError, KeyError):
        return False


def grade(task, agent_sb, image=DEFAULT_IMAGE):
    """Run the check in a fresh container: pristine inputs from a re-run of the (deterministic) setup, plus only
    the task's declared outputs copied from the agent's container. The agent is root in its own container, so
    anything there (input data, the python binary) could have been rigged to fool a check run in place."""
    with Sandbox(image=task.get("image", image)) as judge:
        if task.get("setup", "").strip():
            rc, out = judge.exec(task["setup"])
            if rc:
                raise RuntimeError(f"setup failed in grader for {task['id']}: {out}")
        copied = [p for p in task["outputs"] if agent_sb.copy_to(judge, p)]
        rc, out = judge.exec(task["check"], timeout=CHECK_TIMEOUT_S)
    return rc, out, copied


def run_one(task, model_name, seed, out_dir, config=BASELINE, image=DEFAULT_IMAGE, max_steps=30):
    run_id = f"{config_name(config)}__{task['id']}__s{seed}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.jsonl"
    if _finished(path):
        end = json.loads(path.read_text().splitlines()[-1])
        return {**{k: v for k, v in end.items() if k not in ("ts", "event")}, "trace": str(path), "skipped": True}
    t0 = time.monotonic()
    with path.open("w") as f, Sandbox(image=task.get("image", image)) as sb:
        def trace(event, **kw):
            f.write(json.dumps({"ts": time.time(), "event": event, **kw}, default=str) + "\n")
            f.flush()

        trace("run_start", run_id=run_id, config=config, task=task["id"], category=task.get("category"),
              seed=seed, model=model_name, image=task.get("image", image), max_steps=max_steps)
        if task.get("setup", "").strip():
            rc, out = sb.exec(task["setup"])
            trace("setup", exit_code=rc, output=out)
            if rc:
                raise RuntimeError(f"setup failed for {task['id']}: {out}")
        try:
            result = run_agent(make_model(model_name, task), sb, task["instruction"], trace, build(config), max_steps)
        except Exception as e:  # an API/infra error is a failed run, not a crashed experiment
            result = {"outcome": f"error: {type(e).__name__}: {e}", "steps": None, "usage": {}, "final": ""}
        rc, out, copied = grade(task, sb, image)
        trace("check", exit_code=rc, output=out, outputs_copied=copied)
        summary = {"run_id": run_id, "success": rc == 0, "wall_s": round(time.monotonic() - t0, 2), **result}
        trace("run_end", **summary)
    return {**summary, "trace": str(path)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="claude-haiku-5-5", help="oracle | null | an Anthropic model id")
    p.add_argument("--tasks", default="smoke", help="suite name in hlab/tasks/ or a .toml path")
    p.add_argument("--seeds", type=int, default=1, help="replicates per (config, task); seed = replicate index")
    p.add_argument("--design", action="store_true", help="run all 16 configs of the fractional factorial")
    p.add_argument("--config", action="append", help="a config like A+B-C-D+E-F-G- (repeatable)")
    p.add_argument("--out", default="traces")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("-j", "--workers", type=int, default=1)
    a = p.parse_args(argv)
    configs = fractional_factorial() if a.design else [parse_config(c) for c in a.config or []] or [BASELINE]
    jobs = [(t, c, s) for s in range(a.seeds) for c in configs for t in load_tasks(a.tasks)]
    with ThreadPoolExecutor(a.workers) as pool:  # docker calls are subprocesses, so threads run them in parallel
        results = list(pool.map(lambda j: run_one(j[0], a.model, j[2], a.out, j[1], a.image, a.max_steps), jobs))
    for r in results:
        print(f"{'PASS' if r['success'] else 'FAIL'}  {r['run_id']:<46} {r['outcome'][:20]:<20} {r['wall_s']:>7}s")
    print(f"{sum(r['success'] for r in results)}/{len(results)} passed; traces in {a.out}/")


if __name__ == "__main__":
    main()
