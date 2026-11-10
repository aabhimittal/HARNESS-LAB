"""The seven harness factors. Each is a Middleware; levels are chosen in hlab/middleware/__init__.py."""
import re
from pathlib import Path

from hlab.core.agent import Middleware, final_text, render, run_agent
from hlab.core.tools import _schema

SKILLS_DIR = Path(__file__).parent.parent / "skills"


class Compaction(Middleware):
    """A. Past `threshold` tokens, restart the conversation from the prompt plus a compacted handoff.

    summarize: the model writes a progress summary. query: keep the most recent chunks plus the older chunks
    that share the most words with the task prompt (no model call). Both start a fresh conversation rather
    than editing history in place, so thinking blocks are never invalidated.
    """

    SUMMARY_SYSTEM = ("Summarize this agent transcript for the agent itself to continue the task: what is done, "
                      "key facts and file paths found, what failed, and what remains. Be concise and concrete.")

    def __init__(self, strategy, threshold=60_000, keep_chars=24_000, recent=6):
        assert strategy in ("summarize", "query")
        self.strategy, self.threshold, self.keep_chars, self.recent = strategy, threshold, keep_chars, recent
        self.progress = None  # previous handoff, carried into the next compaction so nothing is lost twice over

    def before_model(self, run):
        # messages[0] is prompt + previous handoff; the prompt is re-sent verbatim, the handoff rides in chunks.
        chunks = ([f"[earlier progress] {self.progress}"] if self.progress else []) + render(run.messages[1:])
        tokens = sum(map(len, chunks)) // 4  # shortcut: chars/4 estimate, use count_tokens if thresholds matter
        if tokens < self.threshold:
            return
        body = (self._summarize(run, chunks) if self.strategy == "summarize"
                else self._keep(run.prompt, chunks, pinned=1 if self.progress else 0))
        self.progress = body
        run.trace("compact", strategy=self.strategy, tokens_before=tokens, chars_after=len(body))
        run.restart(f"{run.prompt}\n\n<progress-so-far>\n{body}\n</progress-so-far>\n"
                    "Your context was compacted. Continue the task from where you left off.")

    def _summarize(self, run, chunks):
        s = run.call_model(self.SUMMARY_SYSTEM, [{"role": "user", "content": "\n".join(chunks)}], [])
        return final_text(s.content)

    def _keep(self, prompt, chunks, pinned=0):
        """Keep the first `pinned` chunks (the earlier handoff), the last `recent`, and the older chunks that
        share the most words with the prompt, within keep_chars. Every chunk is clipped to 2000 chars."""
        words = lambda t: set(re.findall(r"[a-z0-9_./-]{3,}", t.lower()))
        query = words(prompt)
        old = range(pinned, max(pinned, len(chunks) - self.recent))
        keep, budget = set(range(pinned)) | set(range(old.stop, len(chunks))), self.keep_chars
        score = {i: len(words(chunks[i]) & query) for i in old}
        for i in sorted((i for i in old if score[i]), key=lambda i: -score[i]):  # irrelevant chunks never kept
            cost = min(len(chunks[i]), 2000)
            if cost <= budget:
                keep.add(i)
                budget -= cost
        out, prev = [], -1
        for i in sorted(keep):
            if i != prev + 1:
                out.append("[...]")
            out.append(chunks[i][:2000])
            prev = i
        return "\n".join(out)


class Offload(Middleware):
    """B. Tool output over `limit` chars: head+tail stay in context, the full text goes to a file in the sandbox."""

    def __init__(self, limit=4000, keep=1500):
        self.limit, self.keep, self.n = limit, keep, 0

    def after_tool(self, run, name, args, output):
        if len(output) <= self.limit:
            return output
        self.n += 1
        path = f"/work/.hlab/out/{self.n}.txt"
        try:
            run.sandbox.write_file(path, output)
            note = f"full output saved to {path}; use grep/sed on it to see more"
        except Exception as e:  # a full disk must not kill the run; the model still gets head+tail
            note = f"full output could not be saved ({e})"
        cut = len(output) - 2 * self.keep
        return f"{output[:self.keep]}\n[... {cut} chars omitted; {note} ...]\n{output[-self.keep:]}"


class Planning(Middleware):
    """C. Ask for a plan file up front and keep it updated. It survives compaction and Ralph restarts."""
    NOTE = ("\n\nBefore acting, write a short numbered plan with checkboxes to /work/PLAN.md. Keep it updated "
            "as you work: tick items off and revise it when you learn something. Re-read it if you lose track.")

    def setup(self, run):
        run.prompt += self.NOTE


class Verify(Middleware):
    """D. When the model stops, ask it to verify its work; loop until it reports VERIFIED or rounds run out.

    The harness never runs the hidden task check here: that would leak the grader into the agent.
    """
    MARK = "VERIFIED"
    MSG = ("Before you finish, verify your work: re-read the task, run the relevant commands or tests, and check "
           "every requirement. Fix anything that is wrong. When everything checks out, end your reply with the "
           f"word {MARK}.")

    def __init__(self, max_rounds=2):
        self.max_rounds, self.rounds = max_rounds, 0

    def on_stop(self, run, text):
        if self.MARK in text or self.rounds >= self.max_rounds:
            return False
        self.rounds += 1
        run.messages.append({"role": "user", "content": self.MSG})
        return True


class Ralph(Middleware):
    """E. Ralph loop: if the model stops without the completion phrase, rerun the same prompt in a fresh context.

    The filesystem is the only memory between iterations.
    """
    DONE = "TASK COMPLETE"
    AGAIN = ("\n\nNote: this is a fresh attempt at the same task. Earlier attempts may have left work in /work: "
             "inspect it and continue rather than starting over.")

    def __init__(self, max_iters=3):
        self.max_iters, self.iters = max_iters, 1

    def setup(self, run):
        run.prompt += f"\n\nWhen the task is fully done, include the phrase {self.DONE} in your final reply."

    def on_stop(self, run, text):
        if self.DONE in text or self.iters >= self.max_iters:
            return False
        self.iters += 1
        run.trace("ralph_restart", iteration=self.iters)
        run.restart(run.prompt + self.AGAIN)
        return True


class Subagents(Middleware):
    """F. A `delegate` tool that runs a plain subagent (fresh context, same sandbox) and returns its report."""
    tools = [{"name": "delegate", "strict": True,
              "description": "Hand a self-contained subtask to a subagent. It starts with a fresh context, shares "
                             "this container's files, works until done and returns a short report. Give it "
                             "complete instructions: it cannot see this conversation.",
              "input_schema": _schema(task="Complete instructions for the subtask.")}]

    def __init__(self, max_steps=15):
        self.max_steps = max_steps

    def setup(self, run):
        run.handlers["delegate"] = self._delegate

    def _delegate(self, run, args):
        budget = min(self.max_steps, run.max_steps - run.steps)
        if budget <= 0:
            return "no step budget left for a subagent", True
        sub = run_agent(run.model, run.sandbox, args["task"], lambda event, **kw: run.trace("sub_" + event, **kw),
                        max_steps=budget)
        run.steps += sub["steps"]
        run.usage.update(sub["usage"])
        return f"[subagent {sub['outcome']} after {sub['steps']} steps]\n{sub['final']}", False


def load_skills(directory=SKILLS_DIR):
    """Skill files: '---' frontmatter with name/description lines, then a markdown body."""
    skills = {}
    for path in sorted(Path(directory).glob("*.md")):
        _, head, body = path.read_text().split("---", 2)
        meta = dict(line.split(":", 1) for line in head.strip().splitlines())
        skills[meta["name"].strip()] = (meta["description"].strip(), body.strip())
    return skills


class Skills(Middleware):
    """G. off: every skill body is inlined in the system prompt. on: only name+description, plus load_skill.

    Off is "all inlined" rather than "no skills" so the factor isolates the disclosure mechanism, not the knowledge.
    """

    def __init__(self, progressive, skills=None):
        self.progressive, self.skills = progressive, skills or load_skills()
        self.tools = [{"name": "load_skill", "strict": True,
                       "description": "Read the full text of one of the skills listed in the system prompt.",
                       "input_schema": _schema(name="Skill name.")}] if progressive else []

    def setup(self, run):
        if self.progressive:
            index = "\n".join(f"- {n}: {d}" for n, (d, _) in self.skills.items())
            run.system += f"\n\nSkills you can load with the load_skill tool before relying on them:\n{index}"
            run.handlers["load_skill"] = self._load
        else:
            bodies = "\n\n".join(f"## Skill: {n}\n{b}" for n, (_, b) in self.skills.items())
            run.system += f"\n\nSkills:\n\n{bodies}"

    def _load(self, run, args):
        if args["name"] not in self.skills:
            return f"unknown skill {args['name']!r}; available: {', '.join(self.skills)}", True
        return self.skills[args["name"]][1], False
