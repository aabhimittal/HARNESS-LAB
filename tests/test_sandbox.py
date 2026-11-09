import shutil
import subprocess
import time

import pytest

from hlab.sandbox import Sandbox, SandboxError
from hlab.sandbox.docker import MAX_OUTPUT_BYTES


def _docker_ok():
    return shutil.which("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _docker_ok(), reason="docker daemon not available")


def _exists(cid):
    return subprocess.run(["docker", "inspect", cid], capture_output=True).returncode == 0


def test_container_created_and_removed():
    with Sandbox() as sb:
        cid = sb.id
        assert _exists(cid)
    assert not _exists(cid) and sb.id is None


def test_container_removed_on_exception():
    with pytest.raises(ValueError):
        with Sandbox() as sb:
            cid = sb.id
            raise ValueError
    assert not _exists(cid)


def test_exec_exit_code_and_merged_output():
    with Sandbox() as sb:
        assert sb.exec("echo out; echo err >&2; exit 3") == (3, "out\nerr\n")
        assert sb.exec("pwd") == (0, "/work\n")


def test_exec_timeout_kills_process_in_container():
    with Sandbox() as sb:
        t0 = time.monotonic()
        rc, out = sb.exec("sleep 30", timeout=1)
        assert rc == 124 and "timed out" in out and time.monotonic() - t0 < 15
        procs = "for f in /proc/[0-9]*/cmdline; do tr '\\0' ' ' < $f 2>/dev/null; echo; done"
        assert sb.exec(f"{procs} | grep -c '^sleep 30 $'")[1].strip() == "0"  # killed, not orphaned


def test_output_bounded_in_container():
    with Sandbox() as sb:
        # Endless output must not pile up on the host, including via an inherited fd that skips any in-container pipe.
        for cmd in ("yes", "yes >&2", "yes > /proc/$PPID/fd/2"):
            rc, out = sb.exec(cmd, timeout=20)
            assert rc == 141 and len(out) <= MAX_OUTPUT_BYTES + 100, cmd
        sb.exec("head -c 3000000 /dev/zero > /work/big")
        assert len(sb.read_file("/work/big")) == MAX_OUTPUT_BYTES


def test_no_network():
    with Sandbox() as sb:
        assert sb.exec("ls /sys/class/net")[1].split() == ["lo"]


def test_file_roundtrip_and_missing_file():
    content = "line 1\n'quotes' \"and\" $dollars `ticks`\n"
    with Sandbox() as sb:
        sb.write_file("/work/a b/c.txt", content)
        assert sb.read_file("/work/a b/c.txt") == content
        with pytest.raises(SandboxError):
            sb.read_file("/work/nope.txt")


def test_runs_are_isolated():
    with Sandbox() as sb:
        sb.write_file("/work/x", "1")
    with Sandbox() as sb:
        assert sb.exec("test -e /work/x")[0] == 1
