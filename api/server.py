"""
Minimal FastAPI wrapper exposing Argus over HTTP with streamed progress
events, so a frontend (see ui/streamlit_app.py) or any other client can
drive research sessions without importing the Python package directly.

Run with:
    pip install -e .
    uvicorn api.server:app --reload
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from argus.config import settings, validate_settings
from argus.graph import build_graph
from argus.llm import RateLimitedGemini

# Shared, per-process infrastructure — constructed once in `lifespan` below,
# not per-request. `_store` is a single InMemoryStore shared by every user
# in this demo; a real multi-user deployment would still share one Store
# instance (that's the point of `Store` namespacing by user_id) but back it
# with PostgresStore instead so memories survive a restart.
_llm: RateLimitedGemini | None = None
_store = InMemoryStore()
_checkpointer_cm = None
_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _checkpointer_cm, _checkpointer
    validate_settings()
    _llm = RateLimitedGemini()
    _checkpointer_cm = AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path)
    _checkpointer = await _checkpointer_cm.__aenter__()
    yield
    await _checkpointer_cm.__aexit__(None, None, None)

app = FastAPI(title="Argus Research Agent API", lifespan=lifespan)

def _get_graph():
    return build_graph(llm=_llm, checkpointer=_checkpointer, store=_store, max_subquestions=settings.max_subquestions)

class ResearchRequest(BaseModel):
    question: str
    user_id: str = "default_user"
    auto_approve: bool = False


class ResumeRequest(BaseModel):
    thread_id: str
    feedback: str

def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"

async def _stream_until_pause_or_done(graph, stream_input, config: dict):
    """Shared streaming loop for both /research and /research/resume: yields
    a `node_complete` event per finished node, then exactly one of
    `awaiting_human` (graph hit `interrupt()`) or `done` (graph reached
    `finalizer`/END)."""
    interrupt_payload = None
    try:
        async for update in graph.astream(stream_input, config=config, stream_mode="updates"):
            if "__interrupt__" in update:
                interrupt_payload = update["__interrupt__"][0].value
                continue
            node_name = next(iter(update))
            yield _event("node_complete", {"node": node_name})
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client instead of a dropped stream
        yield _event("error", {"message": str(exc)})
        return

    if interrupt_payload is not None:
        yield _event("awaiting_human", interrupt_payload)
    else:
        final_state = await graph.aget_state(config)
        yield _event(
            "done",
            {
                "report_path": final_state.values.get("final_report_path"),
                "report_markdown": final_state.values.get("draft_report_markdown", ""),
            },
        )

@app.post("/research")
async def start_research(req: ResearchRequest):
    thread_id = str(uuid.uuid4())
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "question": req.question,
        "user_id": req.user_id,
        "iteration": 0,
        "max_iterations": settings.max_iterations,
        "auto_approve": req.auto_approve,
    }

    async def event_stream():
        yield _event("started", {"thread_id": thread_id})
        async for chunk in _stream_until_pause_or_done(graph, initial_state, config):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/research/resume")
async def resume_research(req: ResumeRequest):
    graph = _get_graph()
    config = {"configurable": {"thread_id": req.thread_id}}

    async def event_stream():
        async for chunk in _stream_until_pause_or_done(graph, Command(resume=req.feedback), config):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
