"""Factor registry: a config maps each factor letter to -1 (low level) or +1 (high level)."""
from hlab.middleware.factors import Compaction, Offload, Planning, Ralph, Skills, Subagents, Verify

FACTORS = {  # letter: (name, level -1, level +1)
    "A": ("compaction", "summarize", "query-aware keep"),
    "B": ("tool-output offload", "off", "head+tail with file offload"),
    "C": ("planning file", "off", "on"),
    "D": ("self-verification hook", "off", "on"),
    "E": ("Ralph continuation loop", "off", "on"),
    "F": ("subagents", "off", "on"),
    "G": ("skills progressive disclosure", "off (all skills inlined)", "on"),
}


def build(config):
    """Fresh middleware instances for one run. Missing factors default to -1. Order sets on_stop priority."""
    on = lambda f: config.get(f, -1) > 0
    mws = [Compaction("query" if on("A") else "summarize"), Skills(progressive=on("G"))]
    mws += [cls() for f, cls in (("B", Offload), ("C", Planning), ("F", Subagents), ("D", Verify), ("E", Ralph))
            if on(f)]
    return mws


def config_name(config):
    return "".join(f + ("+" if config.get(f, -1) > 0 else "-") for f in FACTORS)
