"""
Researcher node: answers ONE sub-question using web search plus an LLM
call that must ground every claim in a source. This runs as a parallel
branch — LangGraph dispatches one invocation per `research`-kind
sub-question via `Send` (see graph.py), and this function only ever sees
that single sub-question, not the rest of the graph state.
"""
from __future__ import annotations

from ..llm import RateLimitedGemini
from ..schemas import Source, SubFinding, SubQuestion
from ..tools.web_search import search_web

RESEARCH_SYNTHESIS_PROMPT = """Answer this specific question using ONLY the search
results provided below. Every factual claim must be traceable to one of the
sources. If the sources don't cover something, say so explicitly rather
than filling the gap from general knowledge.

Sub-question: {question}

Search results:
{results}

Write a tight, well-organized answer (150-250 words). Do not repeat the
question back at the start.
"""

def researcher_node(state: dict, *, llm: RateLimitedGemini) -> dict:
    # NOTE: `state` here is the `Send` payload (`{"sub_question": {...}}`),
    # not the main `ResearchState` — see graph.py's `fan_out_to_workers` for
    # why that's the case and why it's safe.
    sub_question = SubQuestion.model_validate(state["sub_question"])
    results = search_web(sub_question.question, max_results=5)

    formatted = (
        "\n\n".join(
            f"[{i + 1}] {r['title']}\n{r['content']}\nURL: {r['url']}" for i, r in enumerate(results)
        )
        or "No search results were found."
    )

    prompt = RESEARCH_SYNTHESIS_PROMPT.format(question=sub_question.question, results=formatted)
    summary = llm.invoke(prompt)

    finding = SubFinding(
        sub_question_id=sub_question.id,
        question=sub_question.question,
        summary=summary,
        sources=[Source(url=r["url"], title=r["title"]) for r in results],
        kind="research",
    )
    return {"sub_findings": [finding]}

