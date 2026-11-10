# HARNESS-LAB

Which harness component actually earns its keep? See [SPEC.md](SPEC.md).

## Status

| Milestone | State |
|---|---|
| M0 harness, Docker sandbox, JSONL traces | done |
| M1 seven middlewares, each with tests that its toggle changes behavior | done |
| M2 design matrix generator, alias structure, fold-over, power calculator | done |
| Experiment runner over configs × tasks × seeds, analysis, effect cards | done; tested on oracle runs and planted-effect synthetic data |
| M3 pilot, M4 full run, M5 cross-model | not started: needs model credentials and budget |

## Usage

```bash
pip install -e '.[dev]'
pytest                                       # needs a running docker daemon; ~30s

python -m hlab.design                        # the 16 configs and their alias chains
python -m hlab.design power --tasks 20 40 80 --seeds 1 2 3   # minimum detectable effect table

python -m hlab.run --model oracle --tasks core -j 8          # sanity: every task solvable
python -m hlab.run --design --tasks core --seeds 2 -j 8 --model claude-haiku-5-5   # the experiment
python -m hlab.report traces/ --out report/  # effects.csv + index.html + one effect card per factor
```

The runner skips runs whose trace already ends in `run_end`, so an interrupted sweep resumes. An API error
becomes a failed run, not a crash.

## Layout

- `hlab/core/`: the ReAct loop (`run_agent`), tools, and the `Middleware` hook points
  (`setup`, `before_model`, `after_tool`, `on_stop`).
- `hlab/middleware/`: the seven factors (`factors.py`) and the registry that turns a config into a stack.
- `hlab/sandbox/`: one throwaway container per run. No network, CPU, memory and pid limits, in-container
  timeouts, and host-side output caps.
- `hlab/tasks/`: `smoke.toml` (5 trivial tasks) and `core.toml` (15 tasks built so the factors have something
  to bite on). Each task has `setup`, `check` (exit 0 = success) and a reference `solution`.
- `hlab/skills/`: the skill library factor G discloses.
- `hlab/design.py`, `hlab/analysis.py`, `hlab/report.py`: experiment design, effect estimation, effect cards.

Trace events: `run_start` (config, task, seed, model), `prompt`, `model`, `tool`, `compact`, `ralph_restart`,
`continue`, `sub_*` (subagent), `check`, `run_end` (success, outcome, steps, usage, wall time).
