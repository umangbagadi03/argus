"""
Central place for every environment-driven setting. Nothing else in the
codebase should call `os.environ.get(...)` directly — import `settings`
from here instead, so there's exactly one place that knows where
configuration comes from.

"""
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    google_api_key: str = os.environ.get("GOOGLE_API_KEY", "")
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "")

    # Free-tier engineering knobs. Both matter a lot here: Gemini's free tier
    # is a few requests per minute, and every extra critic->planner loop or
    # extra parallel sub-question is a multiple, not an addition, on total
    # LLM calls per research session.
    max_iterations: int = int(os.environ.get("ARGUS_MAX_ITERATIONS", "2"))
    max_subquestions: int = int(os.environ.get("ARGUS_MAX_SUBQUESTIONS", "4"))

    checkpoint_db_path: str = os.environ.get("ARGUS_CHECKPOINT_DB", ".data/checkpoints.sqlite")
    store_db_path: str = os.environ.get("ARGUS_STORE_DB", ".data/memory.sqlite")

    langsmith_tracing: bool = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"

settings = Settings()


def validate_settings() -> None:
    """Call this at every entrypoint (CLI, API, eval harness) before building
    the graph, so a missing key fails fast with a useful message instead of
    surfacing as a confusing error three calls deep inside LangChain."""
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set.\n"
            "Get a free key (no credit card) at https://aistudio.google.com/apikey "
            "and put it in a .env file — see .env.example."
        )
    if not settings.tavily_api_key:
        # Not fatal: the research/quant nodes degrade to "no search results"
        # rather than crashing (see tools/web_search.py), which is enough to
        # exercise the graph's wiring but not to produce a useful report.
        import warnings

        warnings.warn(
            "TAVILY_API_KEY is not set — research nodes will run with zero "
            "search results. Get a free key (1,000 searches/month, no card) "
            "at https://app.tavily.com to get real answers.",
            stacklevel=2,
        )
