"""
Quant-analyst node: answers ONE sub-question that needs a numeric answer
rather than a qualitative one — e.g. "compare X and Y" — by having the LLM
write a short pandas snippet against context pulled from search, executing
it in a locked-down subprocess (see tools/code_exec.py), and folding the
result back into a `SubFinding`.

It gives the graph an actual computation capability
instead of retrieval-and-summarize only, and it demonstrates routing
different sub-tasks to different specialized workers based on a
classification the planner made.
"""

from __future__ import annotations

from ..llm import RateLimitedGemini
from ..schemas import Source, SubFinding, SubQuestion
from ..tools.code_exec import run_python_snippet
from ..tools.web_search import search_web

CODE_GEN_PROMPT = """You are a data analyst. Given this question and some raw
context gathered from the web, write a short, self-contained Python script
that computes the answer and prints a clean, human-readable result (plain
text or a small markdown table).

Rules:
- Use only the standard library and pandas (already installed).
- Do not read or write files, do not call `input()`, and do not import
  `os`, `sys`, `subprocess`, `socket`, or anything network-related — assume
  no file system or network access is available to the script.
- Hardcode any numbers you need directly from the context below; the
  script cannot fetch anything itself.
- Print the final answer clearly as the last thing the script does.

Question: {question}
Context from web search:
{context}

Return ONLY the Python code. No explanation, no markdown code fences.
"""

SYNTHESIS_PROMPT = """A data-analysis script was run to help answer this
question. Turn its raw output into a short (under 150 words), clearly
written answer suitable for a report section. Mention that it's a computed
result.

Question: {question}
Script output:
{output}
"""

def quant_analyst_node(state: dict, *, llm: RateLimitedGemini) -> dict:
    sub_question = SubQuestion.model_validate(state["sub_question"])
    results = search_web(sub_question.question, max_results=5)
    context = (
        "\n\n".join(f"- {r['title']}: {r['content'][:400]}" for r in results) or "No web context was found."
    )
    raw_code = llm.invoke(CODE_GEN_PROMPT.format(question=sub_question.question, context=context))
    code = _strip_code_fences(raw_code)
    exec_result = run_python_snippet(code, timeout=10)

    if exec_result["timed_out"] or exec_result["returncode"] != 0:
        # Fail soft: a broken generated script should downgrade this
        # sub-question to "unanswered", not crash the whole graph run.
        summary = (
            "A computed answer was attempted for this sub-question, but the "
            f"generated script failed ({(exec_result['stderr'] or 'no output')[:200]}). "
            "Treat this as unanswered rather than guessing."
        )
        table = None
    else:
        summary = llm.invoke(
            SYNTHESIS_PROMPT.format(question=sub_question.question, output=exec_result["stdout"][:2000])
        )
        table = exec_result["stdout"][:2000]

    finding = SubFinding(
        sub_question_id=sub_question.id,
        question=sub_question.question,
        summary=summary,
        sources=[Source(url=r["url"], title=r["title"]) for r in results],
        kind="quant",
        computed_table_markdown=table,
    )
    return {"sub_findings": [finding]}

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text
