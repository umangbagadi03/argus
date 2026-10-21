"""
Critic node: the self-reflection step. Scores the draft against the
original question and the findings that supposedly back it up, and decides
(via the `route_after_critic` conditional edge in graph.py) whether to loop
back for more research or move on to human review.

This is what turns the pipeline from a straight-line
"plan -> research -> write" chain into an actual LangGraph *cycle* —
arguably the single biggest differentiator between this project and a
plain sequential chain of LLM calls.
"""
from __future__ import annotations

from ..llm import RateLimitedGemini
from ..schemas import CritiqueResult, ResearchReport
from ..state import ResearchState
from ..utils import coerce


CRITIQUE_PROMPT = """You are reviewing a research report for quality control.
Be genuinely critical — your job is to find real gaps, not to rubber-stamp
whatever you're given.

Original question: {question}

Report (JSON):
{report_json}

Underlying findings the report was supposed to be grounded in:
{context}

Check for:
1. Every non-obvious claim in the report is actually supported by the
   findings above (not just plausible-sounding).
2. The report actually answers every part of the original question.
3. Citations are present and used correctly — sections that make claims
   have citation_ids pointing at real entries in `citations`.

Return whether it passes, a list of concrete issues (empty if none), a
list of specific follow-up sub-questions that would close the gaps (empty
if passing), and a citation_coverage_score between 0 and 1.
"""

def critic_node(state: ResearchState, *, llm: RateLimitedGemini) -> dict:
    report = coerce(ResearchReport, state["draft_report"])     # coerce to ensure it's a ResearchReport, not just a dict
    prompt = CRITIQUE_PROMPT.format(
        question=state["question"],
        report_json=report.model_dump_json(indent=2),
        context=state.get("compiled_context", ""),
    )

    critique: CritiqueResult = llm.invoke_structured(prompt, CritiqueResult)
    return {"critique": critique}
