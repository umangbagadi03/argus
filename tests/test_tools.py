from argus.nodes.quant_analyst import _strip_code_fences
from argus.tools.code_exec import run_python_snippet


def test_snippet_runs():
    out = run_python_snippet("print(6 * 7)")
    assert out["returncode"] == 0 and out["stdout"].strip() == "42"


def test_snippet_timeout():
    out = run_python_snippet("while True: pass", timeout=1)
    assert out["timed_out"] is True


def test_strip_code_fences():
    assert _strip_code_fences("```python\nprint(1)\n```") == "print(1)"
    assert _strip_code_fences("print(1)") == "print(1)"
