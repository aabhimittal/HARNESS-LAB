"""Minimal ReAct loop with middleware hooks. Each harness factor is a middleware (hlab/middleware/)."""
import json
from collections import Counter
from dataclasses import dataclass, field

from anthropic.types import TextBlock, ToolUseBlock

from hlab.core.tools import TOOLS, execute

SYSTEM = ("You are an autonomous agent in a Linux container. The working directory is /work. "
          "Use the tools to complete the task. When you are finished, reply with a one-line summary "
          "and make no further tool calls.")


@dataclass
class Step:
    content: list  # SDK content blocks; appended to history unchanged (append-only, keeps thinking blocks valid)
    stop_reason: str
    usage: dict = field(default_factory=dict)


class AnthropicModel:
    def __init__(self, model, max_tokens=16000):
        import anthropic
        self.client, self.name, self.max_tokens = anthropic.Anthropic(), model, max_tokens

    def step(self, system, messages, tools):
        kw = {"tools": tools} if tools else {}
        r = self.client.messages.create(model=self.name, max_tokens=self.max_tokens,
                                        system=system, messages=messages, **kw)
        return Step(r.content, r.stop_reason, r.usage.model_dump())


class ScriptedModel:
    """Replays fixed bash commands, then stops. Oracle = the task's reference solution; null = no commands."""

    def __init__(self, commands, name="scripted"):
        self.commands, self.name = list(commands), name

    def step(self, system, messages, tools):
        if self.commands and tools:
            cmd = self.commands.pop(0)
            return Step([ToolUseBlock(type="tool_use", id=f"toolu_{len(messages)}", name="bash",
                                      input={"command": cmd})], "tool_use")
        return Step([TextBlock(type="text", text="done")], "end_turn")


@dataclass
class Run:
    """Mutable state of one agent run; middlewares read and change it through their hooks."""
    model: object
    sandbox: object
    trace: object
    prompt: str  # first user message; middlewares may extend it in setup()
    system: str = SYSTEM
    tools: list = field(default_factory=lambda: list(TOOLS))
    handlers: dict = field(default_factory=dict)  # tool name -> fn(run, args) -> (text, is_error)
    messages: list = field(default_factory=list)
    usage: Counter = field(default_factory=Counter)
    steps: int = 0
    max_steps: int = 30

    def call_model(self, system, messages, tools):
        """Every model call (agent, summarizer, subagent) goes through here so usage is always counted."""
        s = self.model.step(system, messages, tools)
        self.usage.update({k: v for k, v in s.usage.items() if isinstance(v, int)})
        return s

    def restart(self, prompt):
        """Start a fresh conversation. Never edits history in place, so replayed thinking blocks stay valid."""
        self.messages = [{"role": "user", "content": prompt}]


class Middleware:
    """Hook points, in call order. Override only what you need."""
    tools = ()  # extra tool schemas; handle them by registering run.handlers in setup()

    def setup(self, run): ...
    def before_model(self, run): ...
    def after_tool(self, run, name, args, output): return output
    def on_stop(self, run, text): return False  # True = the middleware queued more work; keep looping


def final_text(content):
    return "\n".join(b.text for b in content if b.type == "text")


def render(messages):
    """Flatten a conversation into text chunks (thinking blocks skipped) for summaries and handoffs."""
    chunks = []
    for m in messages:
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for b in content:
            b = b if isinstance(b, dict) else b.model_dump()
            if b["type"] == "text":
                chunks.append(f"[{m['role']}] {b['text']}")
            elif b["type"] == "tool_use":
                chunks.append(f"[tool call: {b['name']}] {json.dumps(b['input'])}")
            elif b["type"] == "tool_result":
                chunks.append(f"[tool result] {b['content']}")
    return chunks


def run_agent(model, sandbox, instruction, trace, middlewares=(), max_steps=30):
    """Run the loop until the model stops and no middleware asks to continue. Returns {outcome, steps, usage}."""
    run = Run(model, sandbox, trace, instruction, max_steps=max_steps)
    for m in middlewares:
        run.tools += list(m.tools)
        m.setup(run)
    run.restart(run.prompt)
    trace("prompt", system=run.system, prompt=run.prompt, tools=[t["name"] for t in run.tools])
    outcome, text = "max_steps", ""
    while run.steps < max_steps:  # subagent steps are charged here too, so loops can't buy free compute
        for m in middlewares:
            m.before_model(run)
        s = run.call_model(run.system, run.messages, run.tools)
        trace("model", step=run.steps, stop_reason=s.stop_reason, usage=s.usage,
              content=[b.model_dump(mode="json", exclude_none=True) for b in s.content])
        run.steps += 1
        run.messages.append({"role": "assistant", "content": s.content})
        if s.stop_reason == "tool_use":
            results = []
            for b in s.content:
                if b.type == "tool_use":
                    handler = run.handlers.get(b.name)
                    out, is_error = handler(run, b.input) if handler else execute(sandbox, b.name, b.input)
                    for m in middlewares:
                        out = m.after_tool(run, b.name, b.input, out)
                    trace("tool", step=run.steps - 1, id=b.id, name=b.name, input=b.input, output=out,
                          is_error=is_error)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out,
                                    "is_error": is_error})
            run.messages.append({"role": "user", "content": results})
            continue
        text = final_text(s.content)
        if s.stop_reason == "end_turn":
            # First middleware that queues more work wins, so list order is priority order (verify before Ralph).
            by = next((type(m).__name__ for m in middlewares if m.on_stop(run, text)), None)
            if by:
                trace("continue", by=by)
                continue
        outcome = s.stop_reason  # end_turn, max_tokens, refusal
        break
    return {"outcome": outcome, "steps": run.steps, "usage": dict(run.usage), "final": text}
