# HARNESS-LAB — which harness component actually earns its keep?

## 1. Thesis

"Agent = model + harness", and harness changes alone can move benchmark rankings dramatically. But nobody publishes component-level causal attribution: how much does compaction strategy, a planning file, a self-verification hook or a Ralph loop contribute, and which components interact? HARNESS-LAB is a modular harness plus a fractional factorial experiment design that answers that with effect sizes and confidence intervals.

Analogy: a Formula 1 team's wind tunnel. You don't redesign the whole car and hope; you change one wing element at a time (or in a designed combination) and measure the downforce each change buys.

## 2. Why now

- LangChain reports moving its coding agent from Top 30 to Top 5 on Terminal-Bench 2.0 by changing only the harness, and notes the same frontier model scores very differently across harnesses (https://www.langchain.com/blog/the-anatomy-of-an-agent-harness).
- The same post names "agents that analyze their own traces to identify and fix harness-level failure modes" as an open problem.
- Harness components are now well-defined enough to toggle: compaction, tool-output offloading, skills disclosure, planning files, verification hooks, Ralph loops, subagents.

## 3. Factors (each a toggle or small set of levels)

| Factor | Levels |
|---|---|
| A Compaction | summarize / query-aware keep |
| B Tool-output offload | off / head+tail with file offload |
| C Planning file | off / on |
| D Self-verification hook (run tests, loop on failure) | off / on |
| E Ralph continuation loop | off / on |
| F Subagents for independent subtasks | off / on |
| G Skills progressive disclosure | off / on |

## 4. Core method

- 7 binary factors → full factorial 2^7 = 128 configs. Use a resolution IV fractional factorial 2^(7−3) = 16 configs: main effects are unconfounded with two-factor interactions.
- Response: task success (binary) per (config, task, seed). Fit a mixed-effects logistic model:
  `logit P(success) = β0 + Σ_j β_j x_j + Σ_{j<k} β_jk x_j x_k + u_task`
  with a random intercept per task (tasks differ in difficulty far more than configs do).
- Secondary responses: tokens, wall time, cost. Report effects with CIs; follow up with a full-factorial fold-over only on the factors that matter.

Power check before spending money: with T tasks, S seeds and 16 configs, estimate the minimum detectable effect on success rate; if it's larger than plausible effects, increase T rather than S.

## 5. Scope

- MVP: modular Python harness + 40-task subset (Terminal-Bench subset or a repo-specific suite) + one cheap model.
- v1: second model to test whether effects transfer across models (the co-training-with-harness question).
- Non-goals: beating leaderboards.

## 6. Architecture

```
harness core (ReAct loop) + middleware stack (each factor = a middleware with on/off)
        ▼
experiment runner (config matrix × tasks × seeds) in sandboxes (Docker)
        ▼
trace store (JSONL per run) ─▶ analysis (statsmodels mixed model) ─▶ effect cards (HTML)
```

"Effect card": one page per factor showing its main effect, top interactions, cost delta, and example traces where it helped and hurt.

## 7. Milestones

- **M0** Minimal harness with bash + file tools in Docker sandbox; deterministic task runner. Acceptance: 5 tasks run end to end with traces.
- **M1** All 7 middlewares implemented with tests that each toggle actually changes behavior.
- **M2** Design matrix generator + power calculator.
- **M3** Pilot run (16 configs × 10 tasks × 2 seeds); sanity-check variance.
- **M4** Full run + mixed-model analysis + effect cards.
- **M5** Cross-model transfer study.

## 8. Repo layout & stack

`hlab/ {core/, middleware/, sandbox/, tasks/, design/, analysis/, report/}`; Python, Docker, statsmodels, pandas.

## 9. Risks & honest weaknesses

- Cost: 16 × 40 × 3 = 1,920 agent runs; budget and choose the model accordingly.
- Your implementation of a component may be worse than a production one, so effects measure your variant.
- Task-set choice drives conclusions; report results per task category.

## 10. Verify before building

Terminal-Bench task licensing and harness interface; sandbox resource limits; model pricing.
