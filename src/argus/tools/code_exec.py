"""
Locked-down local execution of LLM-generated Python snippets, used by the
`quant_analyst` node to actually compute an answer instead of just
summarizing prose.

READ THIS BEFORE REUSING IT ANYWHERE ELSE:
This gives *process* isolation — a fresh interpreter, Python's isolated
mode (`-I`, which ignores `PYTHONPATH`/`PYTHONHOME` and the calling user's
site-packages and drops the script's directory from `sys.path`), and a
hard wall-clock timeout. That's it. It is fine for running your own
LLM-generated snippets, on your own machine, against your own data, for a
portfolio project. It is NOT a security boundary against adversarial code:
a determined attacker with `timeout` seconds can still try to exhaust
memory or disk, or look for a sandbox-escape bug in CPython itself. Real
agent platforms run untrusted generated code in throwaway microVMs or a
hosted sandbox (E2B, Modal Sandboxes, gVisor-backed containers) — swap this
module out for one of those before any untrusted input can reach it.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

def run_python_snippet(code: str, timeout: int = 10) -> dict:
    """Runs `code` in a fresh, isolated Python subprocess and captures
    stdout/stderr. Never raises for a failing snippet — callers should check
    `returncode` / `timed_out` and fail soft (see quant_analyst.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "snippet.py"
        script_path.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout or "",
                "stderr": (exc.stderr or "") + "\n[snippet exceeded the time limit and was killed]",
                "returncode": -1,
                "timed_out": True,
            }
