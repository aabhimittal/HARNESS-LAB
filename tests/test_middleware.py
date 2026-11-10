"""M1 acceptance: each factor's toggle changes what the model sees or what the harness does."""
from anthropic.types import TextBlock, ToolUseBlock

from hlab.core.agent import Step, run_agent, render
from hlab.middleware import FACTORS, build, config_name
from hlab.middleware.factors import Compaction, Offload, Ralph, Skills, Subagents, Verify, load_skills


def text(t):
    return Step([TextBlock(type="text", text=t)], "end_turn")


def tool(tool_name, **args):
    return Step([ToolUseBlock(type="tool_use", id=f"toolu_{tool_name}_{len(args)}", name=tool_name, input=args)],
                "tool_use")


class Recorder:
    """Plays scripted steps (then says 'done') and records every request it receives."""

    def __init__(self, *steps):
        self.steps, self.calls = list(steps), []

    def step(self, system, messages, tools):
        self.calls.append({"system": system, "messages": list(messages), "tools": [t["name"] for t in tools]})
        return self.steps.pop(0) if self.steps else text("done")


class FakeSandbox:
    def __init__(self, output=""):
        self.output, self.files, self.cmds = output, {}, []

    def exec(self, command, timeout=120):
        self.cmds.append(command)
        return 0, self.output

    def write_file(self, path, content):
        self.files[path] = content

    def read_file(self, path):
        return self.files[path]


def go(middlewares, *steps, output="", prompt="fix calc.py so the tests pass", max_steps=30):
    model, sb, events = Recorder(*steps), FakeSandbox(output), []
    result = run_agent(model, sb, prompt, lambda e, **kw: events.append((e, kw)), middlewares, max_steps)
    return model, sb, events, result


def tool_results(model):
    """Tool result texts in the last request the model received."""
    return [b["content"] for m in model.calls[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"]]


def test_config_names_and_baseline():
    assert config_name({"A": 1, "C": 1}) == "A+B-C+D-E-F-G-"
    model, *_ = go(build({}))
    assert model.calls[0]["tools"] == ["bash", "read_file", "write_file"]
    assert "## Skill: python-debugging" in model.calls[0]["system"] and len(FACTORS) == 7


def test_A_compaction_levels_differ_and_restart_fresh():
    big = "calc.py line\n" * 400  # ~5k chars per tool result
    steps = [tool("bash", command="cat calc.py"), tool("bash", command="cat log"), text("done")]
    summ, _, ev_s, _ = go([Compaction("summarize", threshold=1000)], *steps, output=big)
    query, _, ev_q, _ = go([Compaction("query", threshold=1000, keep_chars=500)], *steps, output=big)
    # summarize asks the model (a tool-less call); query-aware keep does not.
    assert [] in [c["tools"] for c in summ.calls] and [] not in [c["tools"] for c in query.calls]
    for model, ev in ((summ, ev_s), (query, ev_q)):
        assert any(e == "compact" for e, _ in ev)
        after = next(c for c in model.calls if c["tools"] and "<progress-so-far>" in str(c["messages"][0]))
        assert len(after["messages"]) == 1  # a fresh conversation, not an edited one


def test_A_compaction_carries_earlier_progress():
    c = Compaction("query", threshold=10, keep_chars=100, recent=1)
    steps = [tool("bash", command="step one"), tool("bash", command="step two"), tool("bash", command="x")]
    go([c], *steps, output="o" * 100)
    assert "earlier progress" in c.progress  # the second handoff includes the first


def test_A_query_keep_prefers_chunks_matching_the_prompt():
    chunks = ["[tool result] unrelated noise"] * 5 + ["[tool result] calc.py has the bug"] + ["[r] recent"] * 6
    kept = Compaction("query", keep_chars=40)._keep("fix calc.py", chunks)
    assert "calc.py has the bug" in kept and "unrelated" not in kept


def test_B_offload():
    big = "x" * 10_000
    on, sb, _, _ = go([Offload()], tool("bash", command="cat big"), output=big)
    off, *_ = go([], tool("bash", command="cat big"), output=big)
    assert len(tool_results(on)[0]) < 4000 and "/work/.hlab/out/1.txt" in tool_results(on)[0]
    assert sb.files["/work/.hlab/out/1.txt"].startswith(big)
    assert len(tool_results(off)[0]) > 10_000


def test_C_planning():
    on, *_ = go(build({"C": 1}))
    off, *_ = go(build({"C": -1}))
    assert "/work/PLAN.md" in on.calls[0]["messages"][0]["content"]
    assert "PLAN.md" not in off.calls[0]["messages"][0]["content"]


def test_D_verify_loops_until_verified_or_rounds_run_out():
    model, *_ = go([Verify()], text("done"), text("checked, VERIFIED"))
    assert len(model.calls) == 2 and "verify your work" in model.calls[1]["messages"][-1]["content"]
    model, *_ = go([Verify(max_rounds=2)])  # never says VERIFIED
    assert len(model.calls) == 3
    model, *_ = go([])
    assert len(model.calls) == 1


def test_E_ralph_restarts_in_fresh_context():
    model, _, events, _ = go([Ralph(max_iters=3)])
    assert len(model.calls) == 3 and all(len(c["messages"]) == 1 for c in model.calls)
    assert "fresh attempt" in model.calls[1]["messages"][0]["content"]
    assert sum(e == "ralph_restart" for e, _ in events) == 2
    model, *_ = go([Ralph()], text("all good. TASK COMPLETE"))
    assert len(model.calls) == 1


def test_D_runs_before_E():
    model, *_ = go(build({"D": 1, "E": 1}), text("done"), text("VERIFIED"), text("ok TASK COMPLETE"))
    assert "verify your work" in model.calls[1]["messages"][-1]["content"]
    assert "fresh attempt" in model.calls[2]["messages"][0]["content"]


def test_F_subagents_share_budget():
    model, _, events, result = go([Subagents()], tool("delegate", task="write primes"), text("sub report"),
                                  text("parent done"))
    assert "delegate" in model.calls[0]["tools"] and "delegate" not in model.calls[1]["tools"]
    assert model.calls[1]["messages"][0]["content"] == "write primes"  # subagent starts from scratch
    assert "sub report" in tool_results(model)[0]
    assert result["steps"] == 3 and any(e == "sub_model" for e, _ in events)
    off, *_ = go(build({}))
    assert "delegate" not in off.calls[0]["tools"]


def test_G_progressive_disclosure():
    skills = load_skills()
    body = skills["python-debugging"][1]
    on, *_ = go([Skills(progressive=True)], tool("load_skill", name="python-debugging"),
                tool("load_skill", name="nope"))
    off, *_ = go([Skills(progressive=False)])
    assert "load_skill" in on.calls[0]["tools"] and body not in on.calls[0]["system"]
    assert "- python-debugging:" in on.calls[0]["system"]
    results = tool_results(on)
    assert results[0] == body and "unknown skill" in results[1]
    assert body in off.calls[0]["system"] and "load_skill" not in off.calls[0]["tools"]


def test_every_factor_changes_the_first_request_or_the_loop():
    """Guard for future factors: flipping any one factor from the baseline changes the transcript."""
    # Enough output (3 x 100k chars, the tool cap) to pass the default compaction threshold, because both
    # levels of A are compaction strategies and only differ once compaction fires.
    steps = lambda: [tool("bash", command=f"cat big{i}") for i in range(3)] + [text("done")]
    base, *_ = go(build({}), *steps(), output="calc " * 30_000)
    for f in FACTORS:
        flipped, *_ = go(build({f: 1}), *steps(), output="calc " * 30_000)
        assert render(flipped.calls[-1]["messages"]) + [flipped.calls[0]["system"], str(flipped.calls[0]["tools"])] != \
            render(base.calls[-1]["messages"]) + [base.calls[0]["system"], str(base.calls[0]["tools"])], f
