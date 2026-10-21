"""
Aggregator node: this is the join point. LangGraph waits for every parallel
`researcher` / `quant_analyst` branch dispatched by the current planning
cycle to finish before it runs this node — you don't have to write any
synchronization yourself, it falls out of both edges (`researcher ->
aggregator`, `quant_analyst -> aggregator`) pointing at the same
downstream node. This node just de-duplicates sources and renders one
compiled context block for `writer`.
"""

from __future__ import annotations

from ..schemas import SubFinding
from ..state import ResearchState
from ..utils import coerce

def aggregator_node(state: ResearchState) -> dict:
    raw_findings = state.get("sub_findings", [])
    findings = [coerce(SubFinding, f) for f in raw_findings]

    blocks = []
    for f in findings:
        urls = [s.url for s in f.sources]
        block = f"### {f.question}\n{f.summary}\n"
        if f.computed_table_markdown:
            block += f"\nComputed output:\n```\n{f.computed_table_markdown}\n```\n"
        block += "Sources: " + (", ".join(urls) if urls else "none")
        blocks.append(block)

    return {"compiled_context": "\n\n".join(blocks)}