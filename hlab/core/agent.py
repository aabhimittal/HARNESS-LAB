"""Minimal ReAct loop. The harness owns the loop so M1 middlewares can hook between model and tool calls."""
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
        r = self.client.messages.create(model=self.name, max_tokens=self.max_tokens,
                                        system=system, tools=tools, messages=messages)
        return Step(r.content, r.stop_reason, r.usage.model_dump())


class ScriptedModel:
    """Replays fixed bash commands, then stops. Oracle = the task's reference solution; null = no commands."""

    def __init__(self, commands, name="scripted"):
        self.commands, self.name = list(commands), name

    def step(self, system, messages, tools):
        if self.commands:
            cmd = self.commands.pop(0)
            return Step([ToolUseBlock(type="tool_use", id=f"toolu_{len(messages)}", name="bash",
                                      input={"command": cmd})], "tool_use")
        return Step([TextBlock(type="text", text="done")], "end_turn")


def run_agent(model, sandbox, instruction, trace, max_steps=30):
    """Run the loop until the model stops calling tools. Returns {outcome, steps, usage}."""
    messages = [{"role": "user", "content": instruction}]
    usage = Counter()
    for i in range(max_steps):
        s = model.step(SYSTEM, messages, TOOLS)
        usage.update({k: v for k, v in s.usage.items() if isinstance(v, int)})
        trace("model", step=i, stop_reason=s.stop_reason, usage=s.usage,
              content=[b.model_dump(mode="json", exclude_none=True) for b in s.content])
        messages.append({"role": "assistant", "content": s.content})
        if s.stop_reason != "tool_use":  # end_turn, max_tokens, refusal: the run is over
            return {"outcome": s.stop_reason, "steps": i + 1, "usage": dict(usage)}
        results = []
        for b in s.content:
            if b.type == "tool_use":
                out, is_error = execute(sandbox, b.name, b.input)
                trace("tool", step=i, id=b.id, name=b.name, input=b.input, output=out, is_error=is_error)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    return {"outcome": "max_steps", "steps": max_steps, "usage": dict(usage)}
