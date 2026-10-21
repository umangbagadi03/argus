"""
Human-in-the-loop checkpoint. Pauses the graph with `interrupt()` and
surfaces the draft report plus the critic's notes for a person to approve
or redirect. This REQUIRES a checkpointer to be compiled into the graph
(see graph.py / main.py) — without one, `interrupt()` has no durable place
to save the paused state and resuming would have nothing to resume from.

In `auto_approve` mode — used by evaluation/run_eval.py to score the
pipeline unattended over a benchmark set — this node skips the interrupt
entirely and answers "approve" on the human's behalf.
"""
from __future__ import annotations

from langgraph.types import interrupt

from ..schemas import CritiqueResult
from ..state import ResearchState
from ..utils import coerce

APPROVE_KEYWORDS = {"approve", "approved", "yes", "looks good", "lgtm", "ok", "okay"}

def human_review_node(state: ResearchState) -> dict:
    if state.get("auto_approve"):
        return {"human_feedback": "approve"}

    critique = coerce(CritiqueResult, state["critique"]) if state.get("critique") else None

    feedback = interrupt(
        {
            "type": "human_review_requested",
            "question": state["question"],
            "draft_report_markdown": state.get("draft_report_markdown", ""),
            "critique": critique.model_dump() if critique else None,
            "instruction": "Reply 'approve' to finalize, or describe what to change.",
        }
    )
    return {"human_feedback": str(feedback).strip()}


def is_approved(feedback: str) -> bool:
    return feedback.strip().lower() in APPROVE_KEYWORDS
