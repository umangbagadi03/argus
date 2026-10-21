"""
Planner node: turns the user's question into a small set of
parallelizable sub-questions, each tagged as `research` (needs web
search + qualitative synthesis) or `quant` (needs a numeric computation).

On the first pass it decomposes the raw question. On later passes — when
this node is re-entered because `critic` sent the graph back with
`follow_up_questions` — it plans ONLY gap-filling sub-questions instead of
re-planning from scratch. That distinction is what keeps the
critic -> planner self-correction loop cheap on a rate-limited free tier:
a second iteration costs a few follow-up calls, not a whole second research
pass.
"""
from __future__ import annotations
from ..llm import RateLimitedGemini
from ..schemas import Plan
from ..state import ResearchState
from ..utils import coerce
from ..schemas import CritiqueResult


PLANNER_PROMPT = """You are a research planning agent. Break the user's question into
{n} or fewer independent, parallelizable sub-questions that together would let
someone write a thorough, well-sourced answer.

For each sub-question, decide whether answering it needs:
- "research": web search and qualitative synthesis, or
- "quant": a numeric computation or comparison over structured data
  (e.g. comparing figures, computing a growth rate, building a small table).

Question: {question}
{context}

Return between 1 and {n} sub-questions. Keep each one specific and
answerable on its own, without needing the others first.
"""

def planner_node(state: ResearchState, *, llm: RateLimitedGemini, max_subquestions: int) -> dict:
    is_replan = state.get("critique") is not None  # if yes then we are re-planning based on critique feedback otherwise we are planning for the first time
    context = ""   

    if is_replan:
        critique = coerce(CritiqueResult, state["critique"])
        gaps = "\n".join(f"- {q}" for q in critique.follow_up_questions)
        context = f"\nThe previous draft had gaps. Focus ONLY on closing these follow-ups:\n{gaps}"

    prompt = PLANNER_PROMPT.format(n=max_subquestions, question=state["question"], context=context)
    plan: Plan = llm.invoke_structured(prompt, Plan)

    # Defensive cap: regardless of what the model returns, never fan out to
    # more sub-questions than configured — each one is a separate parallel
    # LLM call, and free-tier RPM is the tightest resource in this whole
    # system.
    sub_questions = plan.sub_questions[:max_subquestions]

    # IDs are assigned here, not trusted from the model, so `sub_findings`
    # can always be joined back to the sub-question that produced them.
    iteration = state.get("iteration", 0)
    for i, sq in enumerate(sub_questions):
        sq.id = f"sq-{iteration}-{i}"

    updates: dict = {"plan": sub_questions}
    if is_replan:
        updates["iteration"] = iteration + 1
    return updates