"""
Every piece of structured data that crosses an LLM boundary in this project
is a Pydantic model, forced out of the model via `with_structured_output`
rather than parsed out of free-form prose. This is what makes the writer,
critic, and planner nodes reliable enough to route on programmatically —
`critic_node` deciding whether to loop back to `planner` only works because
`CritiqueResult.passed` is guaranteed to be an actual bool, not text you
have to sniff for "looks like a yes".
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

class SubQuestion(BaseModel):
    id: str = ""
    question: str
    kind: Literal["research", "quant"] = "research"

class Plan(BaseModel):
    sub_questions: list[SubQuestion]

class Source(BaseModel):
    url: str
    title: str = ""

# This is what a worker hands back after answering one small question
class SubFinding(BaseModel):
    sub_question_id: str
    question: str
    summary: str
    sources: list[Source] = Field(default_factory=list)
    kind: Literal["research", "quant"] = "research"
    computed_table_markdown: Optional[str] = None

class CritiqueResult(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    citation_coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)

# ReportSection, Citation, and ResearchReport — the final report’s shape
class ReportSection(BaseModel):
    heading: str
    content: str
    citation_ids: list[int] = Field(default_factory=list)

class Citation(BaseModel):
    id: int
    url: str
    title: str = ""

class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    sections: list[ReportSection]
    citations: list[Citation]
    limitations: str
    confidence: Literal["low", "medium", "high"] = "medium"

