from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from argus.graph import build_graph
from argus.memory import get_recent_topics
from argus.nodes.human_review import is_approved

from .conftest import FakeLLM


def _initial(question="What is X?", auto_approve=False, max_iterations=2):
    return {
        "question": question,
        "user_id": "tester",
        "iteration": 0,
        "max_iterations": max_iterations,
        "auto_approve": auto_approve,
    }


def test_auto_approve_runs_end_to_end(tmp_outputs):
    store = InMemoryStore()
    graph = build_graph(llm=FakeLLM(), checkpointer=InMemorySaver(), store=store)
    result = graph.invoke(_initial(auto_approve=True), config={"configurable": {"thread_id": "t1"}})

    assert result["final_report_path"].endswith("Test_Report.md")
    assert len(result["sub_findings"]) == 2
    assert {f.kind for f in result["sub_findings"]} == {"research", "quant"}
    assert get_recent_topics(store, "tester")[0]["title"] == "Test Report"


def test_human_review_interrupt_and_resume(tmp_outputs):
    graph = build_graph(llm=FakeLLM(), checkpointer=InMemorySaver(), store=InMemoryStore())
    config = {"configurable": {"thread_id": "t2"}}

    paused = graph.invoke(_initial(), config=config)
    assert "__interrupt__" in paused
    assert "# Test Report" in paused["__interrupt__"][0].value["draft_report_markdown"]

    # Requesting changes loops back through writer -> critic -> human_review.
    paused_again = graph.invoke(Command(resume="add more detail"), config=config)
    assert "__interrupt__" in paused_again

    done = graph.invoke(Command(resume="approve"), config=config)
    assert done["final_report_path"]


def test_failing_critic_is_bounded_by_max_iterations(tmp_outputs):
    llm = FakeLLM(critic_passes=False)
    graph = build_graph(llm=llm, checkpointer=InMemorySaver(), store=InMemoryStore())
    result = graph.invoke(
        _initial(auto_approve=True, max_iterations=1), config={"configurable": {"thread_id": "t3"}}
    )
    assert result["iteration"] == 1
    assert llm.calls.count("Plan") == 2
    assert result["final_report_path"]


def test_is_approved():
    assert is_approved("  Approve ")
    assert is_approved("lgtm")
    assert not is_approved("please add a section on pricing")
