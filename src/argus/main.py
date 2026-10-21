"""
CLI entrypoint for Argus.

    python -m argus.main "Compare LangGraph and CrewAI adoption in 2026"
    python -m argus.main --thread-id <id> --resume-with "approve"
"""
from __future__ import annotations

import argparse
import sys
import uuid
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from .config import settings, validate_settings
from .graph import build_graph
from .llm import InvalidAPIKeyError, RateLimitedGemini
from .memory import get_recent_topics


def main():
    try:
        _run()
    except InvalidAPIKeyError as exc:
        sys.exit(f"Error: {exc}")


def _run():
    parser = argparse.ArgumentParser(description="Argus — a self-correcting multi-agent research agent")
    parser.add_argument("question", nargs="?", help="Research question to investigate")
    parser.add_argument("--thread-id", default=None, help="Resume an existing (paused) session")
    parser.add_argument("--resume-with", default=None, help="Feedback to resume a paused session with")
    parser.add_argument("--auto-approve", action="store_true", help="Skip human review (mainly for scripting/eval)")
    parser.add_argument("--user-id", default="default_user")
    args = parser.parse_args()

    validate_settings()

    llm = RateLimitedGemini()
    # The long-term Store is in-memory here for simplicity — it only needs
    # to survive the process, unlike the checkpointer below, which needs to
    # survive a paused/resumed session across separate CLI invocations and
    # therefore MUST be a real persistent backend.
    store = InMemoryStore()

    with SqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
        graph = build_graph(llm=llm, checkpointer=checkpointer, store=store, max_subquestions=settings.max_subquestions)

        thread_id = args.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        if not args.thread_id:
            recent = get_recent_topics(store, args.user_id)
            if recent:
                print(
                    f"Welcome back — you've previously researched {len(recent)} topic(s), "
                    f"most recently: '{recent[0]['title']}'\n"
                )

        if args.thread_id and args.resume_with:
            result = graph.invoke(Command(resume=args.resume_with), config=config)
        else:
            if not args.question:
                print("Provide a question, or --thread-id + --resume-with to continue a paused session.")
                sys.exit(1)
            print(f"Starting research session {thread_id}\n")
            result = graph.invoke(
                {
                    "question": args.question,
                    "user_id": args.user_id,
                    "iteration": 0,
                    "max_iterations": settings.max_iterations,
                    "auto_approve": args.auto_approve,
                },
                config=config,
            )

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print("=" * 70)
            print("HUMAN REVIEW REQUESTED")
            print("=" * 70)
            print(payload["draft_report_markdown"])
            print("-" * 70)
            print(payload["instruction"])
            print(f"\nResume with:\n  python -m argus.main --thread-id {thread_id} --resume-with \"approve\"")
        else:
            print(f"\nDone. Report saved to: {result['final_report_path']}")


if __name__ == "__main__":
    main()
