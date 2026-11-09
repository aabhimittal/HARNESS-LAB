"""Tool schemas and dispatch. Every tool executes inside the sandbox, never on the host."""

# shortcut: hard cap so one runaway command can't blow the context window; this is a safety rail,
# not factor B (offload). Keep it far above typical output so the baseline stays "offload off".
MAX_OUTPUT_CHARS = 100_000
BASH_TIMEOUT_S = 120


def _schema(**props):
    return {"type": "object", "properties": {k: {"type": "string", "description": d} for k, d in props.items()},
            "required": list(props), "additionalProperties": False}


TOOLS = [
    {"name": "bash", "strict": True,
     "description": f"Run a bash command in the container (cwd /work). Returns combined stdout/stderr "
                    f"and the exit code. Commands time out after {BASH_TIMEOUT_S}s.",
     "input_schema": _schema(command="The bash command to run.")},
    {"name": "read_file", "strict": True, "description": "Read a text file from the container.",
     "input_schema": _schema(path="Absolute path of the file.")},
    {"name": "write_file", "strict": True,
     "description": "Create or overwrite a text file in the container (parent dirs are created).",
     "input_schema": _schema(path="Absolute path of the file.", content="Full file contents.")},
]


def execute(sandbox, name, args):
    """Run one tool call. Returns (output_text, is_error)."""
    try:
        if name == "bash":
            rc, out = sandbox.exec(args["command"], timeout=BASH_TIMEOUT_S)
            return f"{_cap(out)}\n[exit code {rc}]", False
        elif name == "read_file":
            return _cap(sandbox.read_file(args["path"])), False
        elif name == "write_file":
            sandbox.write_file(args["path"], args["content"])
            return f"wrote {len(args['content'])} chars to {args['path']}", False
        return f"unknown tool: {name}", True
    except Exception as e:  # tool failures go back to the model, never crash the run
        return f"{type(e).__name__}: {e}", True


def _cap(text):
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n[output truncated at {MAX_OUTPUT_CHARS} chars]"
