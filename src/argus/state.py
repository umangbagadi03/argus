"""
The graph's shared State. This is the one type every node reads from and
writes back into (partially — a node only needs to return the keys it's
changing; LangGraph merges the rest in automatically).

The one subtle line here is `sub_findings`. Because it's annotated with
`operator.add`, when several parallel `researcher` / `quant_analyst`
branches (dispatched via `Send`, see graph.py) each return
`{"sub_findings": [one_finding]}`, LangGraph concatenates those lists
together instead of the second write clobbering the first. Every other
field uses the default "last write wins" behavior, which is what you want
for something like `draft_report` — the writer node's whole point is to
replace it.
"""
# — the shared notebook every worker reads and writes

from __future__ import annotations
import operator
from typing import Annotated, Optional, TypedDict
from argus.schemas import CritiqueResult, ResearchReport, SubFinding, SubQuestion

class ResearchState(TypedDict, total=False):
    # Set once at the start of a session.
    question: str
    user_id: str
    auto_approve: bool
    max_iterations: int

    # Written by `planner`, read by the `fan_out_to_workers` conditional edge.
    plan: list[SubQuestion]

    # Written (additively) by `researcher` / `quant_analyst`, read by `aggregator`.
    sub_findings: Annotated[list[SubFinding], operator.add]          # Annotated means add hoge question like list1, list2 na ki overwrite mtlb bas last wala hi add hoga
    compiled_context: str

    # Written by `writer`, read by `critic` / `human_review` / `finalizer`.
    draft_report: Optional[ResearchReport]
    draft_report_markdown: str

    # Written by `critic`, read by the `route_after_critic` conditional edge
    # and surfaced to the human in `human_review`.
    critique: Optional[CritiqueResult]
    iteration: int

    # Written by `human_review`, read by `writer` (as revision instructions)
    # and the `route_after_human` conditional edge.
    human_feedback: Optional[str]

    # Written by `finalizer`.
    final_report: Optional[ResearchReport]
    final_report_path: Optional[str]
