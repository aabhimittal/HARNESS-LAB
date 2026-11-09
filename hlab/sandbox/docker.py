"""Docker sandbox: one throwaway container per agent run, driven through the docker CLI."""
import subprocess

DEFAULT_IMAGE = "python:3.12-slim"
MAX_OUTPUT_BYTES = 1_000_000  # bounded in-container; tools.py trims further for the model


class SandboxError(RuntimeError):
    pass


class Sandbox:
    def __init__(self, image=DEFAULT_IMAGE, cpus=1.0, memory="1g", pids=256, network=False):
        self.image, self.cpus, self.memory, self.pids, self.network = image, cpus, memory, pids, network
        self.id = None

    def start(self):
        args = ["docker", "run", "-d", "--label", "hlab=1", "-w", "/work",
                f"--cpus={self.cpus}", f"--memory={self.memory}", f"--pids-limit={self.pids}"]
        if not self.network:
            args += ["--network", "none"]
        p = subprocess.run([*args, self.image, "sleep", "infinity"], capture_output=True, text=True, timeout=300)
        if p.returncode:
            raise SandboxError(f"docker run failed: {p.stderr.strip()}")
        self.id = p.stdout.strip()
        return self

    def stop(self):
        if self.id:
            subprocess.run(["docker", "rm", "-f", self.id], capture_output=True, timeout=60)
            self.id = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def exec(self, command, timeout=120):
        """Run a bash command; returns (exit_code, combined stdout+stderr). Exit 124 = timed out."""
        # Kill inside the container: a client-side timeout alone would leave the process running there.
        # In-container: merge stderr into stdout (docker exec ships them separately and their order races),
        # and cap bytes with head so a runaway command can't flood host memory. Exit code is the command's own.
        wrapper = f'bash -c "$1" 2>&1 | head -c {MAX_OUTPUT_BYTES}; exit "${{PIPESTATUS[0]}}"'
        args = ["docker", "exec", self.id, "timeout", "-k", "5", str(timeout), "bash", "-c", wrapper, "_", command]
        try:
            p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return 124, f"[timed out after {timeout}s]"
        out = p.stdout.decode(errors="replace")
        if p.returncode == 124:
            out += f"\n[timed out after {timeout}s]"
        return p.returncode, out

    def write_file(self, path, content):
        script = 'mkdir -p "$(dirname "$1")" && cat > "$1"'
        p = subprocess.run(["docker", "exec", "-i", self.id, "sh", "-c", script, "sh", path],
                           input=content.encode(), capture_output=True, timeout=60)
        if p.returncode:
            raise SandboxError(p.stderr.decode(errors="replace").strip())

    def read_file(self, path):
        p = subprocess.run(["docker", "exec", self.id, "head", "-c", str(MAX_OUTPUT_BYTES), "--", path],
                           capture_output=True, timeout=60)
        if p.returncode:
            raise SandboxError(p.stderr.decode(errors="replace").strip())
        return p.stdout.decode(errors="replace")
