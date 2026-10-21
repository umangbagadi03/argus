"""
Thin wrapper around Tavily so the rest of the codebase depends on one
function — `search_web(query) -> list[dict]` — rather than any specific
vendor SDK. Swap the body of this module for Brave Search, SerpAPI, Bing,
or a self-hosted SearxNG instance without touching a single node; they all
just need to return the same `{"title", "url", "content"}` shape.

Tavily was picked over a keyless scraper (e.g. DuckDuckGo scraping
libraries) deliberately: it returns clean, pre-extracted page content
instead of raw HTML, which matters a lot for keeping researcher-node
prompts small on a token-metered free tier, and its free tier (1,000
search credits/month, no card) is stable and officially supported by
LangChain.
"""
from __future__ import annotations
from functools import lru_cache
from langchain_tavily import TavilySearch

from .. import config

@lru_cache(maxsize=1)
def _get_tool() -> TavilySearch:
    return TavilySearch(
        max_results=5,
        topic="general",
        search_depth="basic"
    )

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Returns a list of `{"title", "url", "content"}` dicts, or `[]` if no
    Tavily key is configured (lets the graph run end-to-end for wiring
    demos/tests without a second API key, just with empty findings)."""
    if not config.settings.tavily_api_key:
        return []

    tool = _get_tool()
    raw = tool.invoke({"query": query})

    # Defensive: different `langchain-tavily` versions have returned either
    # the raw Tavily API shape (`{"results": [...], "answer": ...}`) or a
    # bare list of result dicts. Handle both rather than assuming one.
    results = raw.get("results", []) if isinstance(raw, dict) else raw

    out = []
    for r in (results or [])[:max_results]:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content", "") or "")[:1500],
            }
        )
    return out