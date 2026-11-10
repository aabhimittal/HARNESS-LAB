"""Docker sandbox: one throwaway container per agent run, driven through the docker CLI."""
import subprocess
import threading

DEFAULT_IMAGE = "python:3.12-slim"
MAX_OUTPUT_BYTES = 1_000_000  # host-side cap per exec/read; tools.py trims further for the model


class SandboxError(RuntimeError):
    pass


class Sandbox:
    def __init__(self, image=DEFAULT_IMAGE, cpus=1.0, memory="1g", pids=256, network=False):
        self.image, self.cpus, self.memory, self.pids, self.network = image, cpus, memory, pids, network
        self.id = None

    def start(self):
        # --log-driver none: writes to /proc/1/fd/1 would otherwise grow docker's log file on the host disk.
        args = ["docker", "run", "-d", "--label", "hlab=1", "-w", "/work", "--log-driver", "none",
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
        # Merge stderr in-container too: docker exec ships the two streams separately and their order races.
        args = ["docker", "exec", self.id, "timeout", "-k", "5", str(timeout), "bash", "-c", "exec 2>&1\n" + command]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        killer = threading.Timer(timeout + 30, p.kill)  # backstop if docker itself hangs
        killer.start()
        try:
            # Cap on the host: the command can reach other inherited fds, so no in-container limit is airtight.
            data = p.stdout.read(MAX_OUTPUT_BYTES + 1)
            truncated = len(data) > MAX_OUTPUT_BYTES
            if truncated:
                p.kill()
            rc = p.wait()
        finally:
            killer.cancel()
            p.stdout.close()
        out = data[:MAX_OUTPUT_BYTES].decode(errors="replace")
        if truncated:
            return 141, out + f"\n[output exceeded {MAX_OUTPUT_BYTES} bytes; command killed]"
        if rc == 124 or rc < 0:
            return 124, out + f"\n[timed out after {timeout}s]"
        return rc, out

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
