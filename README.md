# HARNESS-LAB

Which harness component actually earns its keep? See [SPEC.md](SPEC.md).

## Status: M0

Minimal ReAct harness (`hlab/core/`) with `bash`, `read_file` and `write_file` tools. Every tool runs inside a
throwaway Docker container (`hlab/sandbox/`): no network, 1 CPU, 1 GB, 256 pids. Each run writes one JSONL trace.

```bash
pip install -e '.[dev]'
pytest                                   # needs a running docker daemon
python -m hlab.run --model oracle        # replay reference solutions: 5/5 should pass
python -m hlab.run --model null          # do nothing: 0/5 should pass (checks aren't vacuous)
python -m hlab.run --model claude-haiku-5-5 --seeds 2   # real agent; needs Anthropic credentials
```

Trace events: `run_start`, `setup`, `model` (content blocks + usage), `tool` (input/output), `check`,
`run_end` (success, outcome, steps, summed usage, wall time).

Tasks live in `hlab/tasks/smoke.toml`. Each has an `instruction`, a `setup` script, a `check` script (exit 0 =
success) and a reference `solution`.
