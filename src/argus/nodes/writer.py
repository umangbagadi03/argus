"""
Writer node: turns the compiled research context into a structured,
citation-carrying `ResearchReport`, forced through `with_structured_output`
rather than hoping the model's prose happens to parse.

If `human_feedback` is present in state (and a draft already exists), this
node treats it as a REVISION pass over the existing draft instead of
writing from scratch. That's what makes the `human_review -> writer` edge
a genuine editing loop rather than a "throw the draft away and start over"
loop — the reviewer's earlier context isn't lost.
"""
from __future__ import annotations

from ..llm import RateLimitedGemini
from ..schemas import ResearchReport
from ..state import ResearchState
from ..utils import coerce

WRITE_PROMPT = """Using ONLY the research findings below, write a structured
report answering the user's original question.

Original question: {question}

Research findings:
{context}

Requirements:
- Every section's claims must be traceable to the findings above.
- Populate `citations` with every source URL you actually used, numbered
  from 1, and reference those numbers in each section's `citation_ids`.
- Be explicit in `limitations` about anything the findings didn't cover.
- Set `confidence` honestly based on how well-supported the findings were.
"""

REVISE_PROMPT = """Revise the report below based on the reviewer's feedback.
Keep everything that wasn't flagged; change only what the feedback asks
for. Keep the same structured schema.

Original question: {question}
Reviewer feedback: {feedback}

Current report (JSON):
{report_json}

Research findings available, in case the feedback asks for detail you can
pull from what's already gathered:
{context}
"""

def writer_node(state: ResearchState, *, llm: RateLimitedGemini) -> dict:
    if state.get("human_feedback") and state.get("draft_report"):  # if human feedback then revise the draft
        prior = coerce(ResearchReport, state["draft_report"])
        prompt = REVISE_PROMPT.format(
            question=state["question"],
            feedback=state["human_feedback"],
            report_json=prior.model_dump_json(indent=2),
            context=state.get("compiled_context", ""),
        )
    else:    # else write a new report from scratch
        prompt = WRITE_PROMPT.format(
            question=state["question"],
            context=state.get("compiled_context", ""),
        )
    report: ResearchReport = llm.invoke_structured(prompt, ResearchReport)
    return {
        "draft_report": report,
        "draft_report_markdown": render_report_markdown(report),
        "human_feedback": None,  # consumed — don't let stale feedback leak into the next loop
    }

def render_report_markdown(report: ResearchReport) -> str:
    lines = [f"# {report.title}", "", f"**Confidence:** {report.confidence}", "", "## Executive Summary", report.executive_summary, ""]
    for section in report.sections:
        cites = " ".join(f"[{i}]" for i in section.citation_ids)
        header = f"## {section.heading} {cites}".rstrip()
        lines.append(header)
        lines.append(section.content)
        lines.append("")
    lines.append("## Limitations")
    lines.append(report.limitations)
    lines.append("")
    lines.append("## Sources")
    for c in report.citations:
        lines.append(f"[{c.id}] {c.title} — {c.url}")
    return "\n".join(lines)
