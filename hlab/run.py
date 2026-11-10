"""Deterministic task runner: fixed task order, fresh container per (task, seed), one JSONL trace per run.

    python -m hlab.run --model oracle            # replay reference solutions (no API key needed)
    python -m hlab.run --model claude-haiku-5-5  # real agent
"""
import argparse
import json
import time
import tomllib
from pathlib import Path

from hlab.core.agent import AnthropicModel, ScriptedModel, run_agent
from hlab.sandbox import DEFAULT_IMAGE, Sandbox

SMOKE = Path(__file__).parent / "tasks" / "smoke.toml"
CHECK_TIMEOUT_S = 300


def load_tasks(path=SMOKE):
    with open(path, "rb") as f:
        return tomllib.load(f)["task"]


def make_model(name, task):
    if name == "oracle":
        return ScriptedModel([task["solution"]], name="oracle")
    if name == "null":
        return ScriptedModel([], name="null")
    return AnthropicModel(name)


def run_one(task, model_name, seed, out_dir, config="baseline", image=DEFAULT_IMAGE, max_steps=30):
    run_id = f"{config}__{task['id']}__s{seed}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.jsonl"
    t0 = time.monotonic()
    with path.open("w") as f, Sandbox(image=image) as sb:
        def trace(event, **kw):
            f.write(json.dumps({"ts": time.time(), "event": event, **kw}, default=str) + "\n")
            f.flush()

        trace("run_start", run_id=run_id, config=config, task=task["id"], category=task.get("category"),
              seed=seed, model=model_name, image=image)
        if task.get("setup", "").strip():
            rc, out = sb.exec(task["setup"])
            trace("setup", exit_code=rc, output=out)
            if rc:
                raise RuntimeError(f"setup failed for {task['id']}: {out}")
        try:
            result = run_agent(make_model(model_name, task), sb, task["instruction"], trace, max_steps)
        except Exception as e:  # an API/infra error is a failed run, not a crashed experiment
            result = {"outcome": f"error: {type(e).__name__}: {e}", "steps": None, "usage": {}}
        rc, out = sb.exec(task["check"], timeout=CHECK_TIMEOUT_S)
        trace("check", exit_code=rc, output=out)
        summary = {"run_id": run_id, "success": rc == 0, "wall_s": round(time.monotonic() - t0, 2), **result}
        trace("run_end", **summary)
    return {**summary, "trace": str(path)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="claude-haiku-5-5", help="oracle | null | an Anthropic model id")
    p.add_argument("--tasks", default=str(SMOKE))
    p.add_argument("--seeds", type=int, default=1, help="replicates per task (seed = replicate index)")
    p.add_argument("--out", default="traces")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--max-steps", type=int, default=30)
    a = p.parse_args(argv)
    results = [run_one(t, a.model, s, a.out, image=a.image, max_steps=a.max_steps)
               for s in range(a.seeds) for t in load_tasks(a.tasks)]
    for r in results:
        print(f"{'PASS' if r['success'] else 'FAIL'}  {r['run_id']:<40} {r['outcome']:<12} {r['wall_s']:>6}s")
    print(f"{sum(r['success'] for r in results)}/{len(results)} passed; traces in {a.out}/")


if __name__ == "__main__":
    main()
