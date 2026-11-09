"""Run a generated script inside the OptimUS Docker image.

The solver needs bempp-legacy on Python 3.5, which only the image has, so
scripts never run on the host. Each run gets a fresh temporary folder, no
network, a time limit, and the non-interactive matplotlib backend so
figures are saved instead of opening windows.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from . import config
from .loop import RunResult


def docker_command(workdir: str, image: str = config.DOCKER_IMAGE, script: str = "script.py",
                   data_dir: Optional[str] = None) -> List[str]:
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "-e", "MPLBACKEND=Agg",
        "-v", "%s:/work" % workdir,
        "-w", "/work",
    ]
    if data_dir:
        # The notebooks' Data folder, read-only, for scripts that load meshes.
        cmd += ["-v", "%s:/work/Data:ro" % data_dir]
    return cmd + [image, "python3", script]


def extract_error(stderr: str, max_lines: int = 30) -> str:
    """The last traceback, trimmed, so a fix prompt stays small."""
    lines = stderr.strip().splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("Traceback (most recent call last)")]
    if starts:
        lines = lines[starts[-1]:]
    if len(lines) > max_lines:
        lines = lines[:3] + ["    ..."] + lines[-(max_lines - 4):]
    return "\n".join(lines)


def run_in_docker(code: str, timeout: int = config.RUN_TIMEOUT_S, image: str = config.DOCKER_IMAGE,
                  data_dir: Optional[str] = None) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="optimus_run_") as tmp:
        Path(tmp, "script.py").write_text(code, encoding="utf-8")
        t0 = time.perf_counter()
        try:
            p = subprocess.run(docker_command(tmp, image, data_dir=data_dir), capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return RunResult(False, error="Timed out after %d s" % timeout, seconds=float(timeout))
        except FileNotFoundError:
            return RunResult(False, error="Docker is not installed or not on PATH")
        secs = time.perf_counter() - t0
        if p.returncode == 0:
            return RunResult(True, output=p.stdout[-4000:], seconds=secs)
        return RunResult(False, output=p.stdout[-4000:], error=extract_error(p.stderr) or p.stdout[-2000:],
                         seconds=secs)
