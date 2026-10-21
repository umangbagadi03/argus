"""
Helpers around the cross-thread long-term-memory `Store`. Kept separate
from `finalizer.py` (which writes) so anything that wants to read a user's
history — today the CLI's "welcome back" message, tomorrow maybe a
personalization step in `planner` — has one obvious place to import from.
"""

from __future__ import annotations
from langgraph.store.base import BaseStore

NAMESPACE_HISTORY = "research_history"
def get_recent_topics(store: BaseStore, user_id: str, limit: int = 5) -> list[dict]:
    items = store.search((user_id, NAMESPACE_HISTORY), limit=limit)
    # Keys are ISO timestamps (see finalizer.py), so sort on them explicitly
    # rather than relying on a particular store implementation's insertion
    # order.
    ordered = sorted(items, key=lambda it: it.key, reverse=True)
    return [it.value for it in ordered[:limit]]
