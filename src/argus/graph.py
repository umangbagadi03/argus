"""
Assembles every node into the Argus StateGraph.

    START -> planner
          -[Send fan-out, parallel]-> { researcher | quant_analyst }
          -> aggregator -> writer -> critic
          -[not passed & iterations left]-> back to planner   (self-correction cycle)
          -[passed, or out of iterations]-> human_review
          -[changes requested]-> back to writer                (revision cycle)
          -[approved]-> finalizer -> END

That's two independent cycles (critic->planner, human_review->writer), one
parallel map-reduce fan-out (`Send`), one interrupt point, one checkpointer,
and one cross-thread `Store`. See the README for the full
rationale and a walkthrough of what each primitive is
doing and why.
"""

from __future__ import annotations
import functools
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Send
from .llm import RateLimitedGemini
from .nodes.aggregator import aggregator_node
from .nodes.critic import critic_node
from .nodes.finalizer import finalizer_node
from .nodes.human_review import human_review_node, is_approved
from .nodes.planner import planner_node
from .nodes.quant_analyst import quant_analyst_node
from .nodes.researcher import researcher_node
from .nodes.writer import writer_node
from .schemas import CritiqueResult, SubQuestion
from .state import ResearchState
from .utils import coerce

#---------------------------------------------------------------------------
# Routing functions. Kept separate from the node functions above on purpose:
# a node computes/writes state, a routing function only *reads* state to
# decide where execution goes next. Mixing the two makes graphs much harder
# to reason about once they have more than two or three nodes.
# ---------------------------------------------------------------------------

def fan_out_to_workers(state: ResearchState) -> list[Send]:
    """Conditional edge out of `planner`. Dispatches one parallel branch per
    sub-question, routed to `researcher` or `quant_analyst` by `kind`. Each
    `Send`'s payload becomes the ENTIRE local state for that branch — it is
    not merged with the main graph state until the branch node returns."""
    sends = []
    for raw_sq in state["plan"]:
        sq = coerce(SubQuestion, raw_sq)
        target = "quant_analyst" if sq.kind == "quant" else "researcher"
        sends.append(Send(target, {"sub_question": sq.model_dump()}))
    return sends

def route_after_critic(state: ResearchState) -> str:
    critique = coerce(CritiqueResult, state["critique"])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)
    if not critique.passed and iteration < max_iterations:
        return "planner"
    return "human_review"


def route_after_human(state: ResearchState) -> str:
    if is_approved(state.get("human_feedback", "")):
        return "finalizer"
    return "writer"

# ---------------------------------------------------------------------------
# Graph construction.
# ---------------------------------------------------------------------------


def build_graph(
    llm: RateLimitedGemini,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    max_subquestions: int = 4,
):
    """
    Builds and compiles the graph.

    Dependency injection note: `llm` and `store` are bound into the node
    functions via `functools.partial` — they're infrastructure, constructed
    once and shared across every request/session. `user_id`, `question`,
    `iteration`, and everything else request-specific flows through graph
    STATE instead, never through partial-binding, because this graph object
    is typically compiled once (see api/server.py) and then reused across
    many different users' sessions — anything bound at build time would be
    shared by every one of them. (Earlier drafts of `finalizer_node` bound
    `user_id` this way; it's a real bug worth knowing to watch for.)
    """
    builder = StateGraph(ResearchState)

    builder.add_node("planner", functools.partial(planner_node, llm=llm, max_subquestions=max_subquestions))
    builder.add_node("researcher", functools.partial(researcher_node, llm=llm))
    builder.add_node("quant_analyst", functools.partial(quant_analyst_node, llm=llm))
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("writer", functools.partial(writer_node, llm=llm))
    builder.add_node("critic", functools.partial(critic_node, llm=llm))
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalizer", functools.partial(finalizer_node, store=store))

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", fan_out_to_workers, ["researcher", "quant_analyst"])
    builder.add_edge("researcher", "aggregator")
    builder.add_edge("quant_analyst", "aggregator")
    builder.add_edge("aggregator", "writer")
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges("critic", route_after_critic, ["planner", "human_review"])
    builder.add_conditional_edges("human_review", route_after_human, ["writer", "finalizer"])
    builder.add_edge("finalizer", END)

    return builder.compile(checkpointer=checkpointer, store=store)
