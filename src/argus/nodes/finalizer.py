"""
Finalizer node: writes the approved report to disk and records a
long-term-memory entry (topic, title, timestamp) in the `Store` so a
FUTURE session — a different `thread_id` entirely, possibly started
tomorrow — can be personalized around what this user has already
researched (see `memory.get_recent_topics`, used by main.py's CLI).

This is deliberately the one node that demonstrates *cross-thread* memory,
as opposed to the checkpointer's *within-thread* persistence used for the
interrupt/resume flow. They solve different problems: the checkpointer
answers "what happened earlier in THIS session"; the Store answers "what
do we know about THIS user across every session they've ever run."
"""
from __future__ import annotations
import datetime as dt
from pathlib import Path
from langgraph.store.base import BaseStore
from ..schemas import ResearchReport
from ..state import ResearchState
from ..utils import coerce

OUTPUT_DIR = Path("outputs")

def finalizer_node(state: ResearchState, *, store: BaseStore) -> dict:
    report = coerce(ResearchReport, state["draft_report"])
    user_id = state.get("user_id") or "default_user"

    OUTPUT_DIR.mkdir(exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in report.title)[:60].strip() or "report"
    path = OUTPUT_DIR / f"{safe_title.replace(' ', '_')}.md"
    path.write_text(state.get("draft_report_markdown", ""))

    store.put(
        (user_id, "research_history"),
        key=dt.datetime.now(dt.timezone.utc).isoformat(),
        value={"question": state["question"], "title": report.title, "confidence": report.confidence},
    )

    return {"final_report": report, "final_report_path": str(path)}
