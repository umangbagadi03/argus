"""
Shared fixtures. `FakeLLM` stands in for `RateLimitedGemini` so the whole
graph can be exercised offline — no API keys, no quota, deterministic.
"""
from __future__ import annotations

import pytest

from argus.schemas import (
    Citation,
    CritiqueResult,
    Plan,
    ReportSection,
    ResearchReport,
    SubQuestion,
)
from argus.tools import web_search


class FakeLLM:
    def __init__(self, critic_passes: bool = True):
        self.critic_passes = critic_passes
        self.calls: list[str] = []

    def invoke(self, prompt: str, **kwargs) -> str:
        self.calls.append("invoke")
        if "Return ONLY the Python code" in prompt:
            return "```python\nprint('| a | b |\\n|---|---|\\n| 1 | 2 |')\n```"
        return "A grounded summary of the findings."

    def invoke_structured(self, prompt: str, schema):
        self.calls.append(schema.__name__)
        if schema is Plan:
            return Plan(
                sub_questions=[
                    SubQuestion(question="What is X?", kind="research"),
                    SubQuestion(question="How big is X?", kind="quant"),
                ]
            )
        if schema is ResearchReport:
            return ResearchReport(
                title="Test Report",
                executive_summary="Summary.",
                sections=[ReportSection(heading="Findings", content="Body.", citation_ids=[1])],
                citations=[Citation(id=1, url="https://example.com", title="Example")],
                limitations="None.",
                confidence="medium",
            )
        if schema is CritiqueResult:
            return CritiqueResult(
                passed=self.critic_passes,
                issues=[] if self.critic_passes else ["Missing detail"],
                follow_up_questions=[] if self.critic_passes else ["What about Y?"],
                citation_coverage_score=0.9,
            )
        raise AssertionError(f"Unexpected schema {schema}")


@pytest.fixture(autouse=True)
def offline_search(monkeypatch):
    monkeypatch.setattr(
        web_search,
        "search_web",
        lambda query, max_results=5: [{"title": "Example", "url": "https://example.com", "content": "X is 42."}],
    )
    # Nodes import `search_web` by name, so patch their references too.
    from argus.nodes import quant_analyst, researcher

    monkeypatch.setattr(researcher, "search_web", web_search.search_web)
    monkeypatch.setattr(quant_analyst, "search_web", web_search.search_web)


@pytest.fixture
def tmp_outputs(monkeypatch, tmp_path):
    from argus.nodes import finalizer

    monkeypatch.setattr(finalizer, "OUTPUT_DIR", tmp_path)
    return tmp_path
