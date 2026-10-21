"""
Unattended evaluation harness. Runs every question in eval_dataset.json
through the full graph with `auto_approve=True`, then asks an LLM judge
whether the final report covers each `must_cover` point.

    python evaluation/run_eval.py
    python evaluation/run_eval.py --only eval-1 eval-3

Results are written to outputs/eval_results.json. Thanks to the disk cache
in llm.py, re-running after a scoring-only change costs no extra quota.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from argus.config import settings, validate_settings
from argus.graph import build_graph
from argus.llm import RateLimitedGemini

DATASET = Path(__file__).parent / "eval_dataset.json"
RESULTS = Path("outputs") / "eval_results.json"

JUDGE_PROMPT = """You are grading a research report. For each required point
below, decide whether the report genuinely covers it (not just mentions a
keyword).

Question: {question}

Required points:
{points}

Report:
{report}
"""


class PointVerdict(BaseModel):
    point: str
    covered: bool
    reason: str = ""


class JudgeResult(BaseModel):
    verdicts: list[PointVerdict] = Field(default_factory=list)


def main():
    parser = argparse.ArgumentParser(description="Run the Argus evaluation set")
    parser.add_argument("--only", nargs="*", help="Run only these eval ids")
    args = parser.parse_args()

    validate_settings()
    llm = RateLimitedGemini()
    graph = build_graph(
        llm=llm, checkpointer=InMemorySaver(), store=InMemoryStore(), max_subquestions=settings.max_subquestions
    )

    cases = json.loads(DATASET.read_text())
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    results = []
    for case in cases:
        print(f"[{case['id']}] {case['question']}")
        state = graph.invoke(
            {
                "question": case["question"],
                "user_id": "eval",
                "iteration": 0,
                "max_iterations": settings.max_iterations,
                "auto_approve": True,
            },
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        judged: JudgeResult = llm.invoke_structured(
            JUDGE_PROMPT.format(
                question=case["question"],
                points="\n".join(f"- {p}" for p in case["must_cover"]),
                report=state.get("draft_report_markdown", ""),
            ),
            JudgeResult,
        )
        covered = sum(v.covered for v in judged.verdicts)
        score = covered / len(case["must_cover"]) if case["must_cover"] else 1.0
        print(f"    coverage: {covered}/{len(case['must_cover'])}  ({score:.0%})")
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "coverage": score,
                "iterations": state.get("iteration", 0),
                "report_path": state.get("final_report_path"),
                "verdicts": [v.model_dump() for v in judged.verdicts],
            }
        )

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    if results:
        mean = sum(r["coverage"] for r in results) / len(results)
        print(f"\nMean coverage: {mean:.0%}  ->  {RESULTS}")


if __name__ == "__main__":
    main()
