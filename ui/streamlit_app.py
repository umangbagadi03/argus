"""
Streamlit UI for Argus. Talks to the FastAPI server (see api/server.py) over
SSE, so it exercises the exact same code path any real frontend would —
nothing here reaches into the `argus` package directly.

Run (two terminals), or use ./run.sh to start both:
    uvicorn api.server:app --reload
    streamlit run ui/streamlit_app.py
"""
import json
import os

import httpx
import streamlit as st

API_BASE = os.environ.get("ARGUS_API_URL", "http://localhost:8000")

# Human-readable labels for each graph node, in pipeline order.
STEPS = {
    "planner": "Planning sub-questions",
    "researcher": "Researching the web",
    "quant_analyst": "Running quantitative analysis",
    "aggregator": "Aggregating findings",
    "writer": "Writing the report",
    "critic": "Critiquing the draft",
    "human_review": "Waiting for your review",
    "finalizer": "Saving the final report",
}

EXAMPLES = [
    "Compare LangGraph, CrewAI, and AutoGen for a production multi-agent system",
    "What are the trade-offs between RAG and long-context LLMs for enterprise search?",
    "How fast is the global EV market growing, and which regions lead adoption?",
]

st.set_page_config(page_title="Argus Research Agent", page_icon="🔎", layout="wide")

for key, default in {
    "thread_id": None,
    "awaiting_human": None,
    "final_report": None,
    "final_path": None,
    "log": [],
    "error": None,
    "question": "",
}.items():
    st.session_state.setdefault(key, default)


def api_online() -> bool:
    try:
        return httpx.get(f"{API_BASE}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def stream_events(url: str, payload: dict):
    """Yields `(event_name, data)` tuples from the server's SSE stream."""
    with httpx.stream("POST", url, json=payload, timeout=None) as resp:
        resp.raise_for_status()
        event_name = "message"
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                yield event_name, json.loads(line[len("data:"):].strip())


def run_stream(url: str, payload: dict):
    st.session_state.error = None
    with st.status("Argus is working…", expanded=True) as status:
        try:
            for name, data in stream_events(url, payload):
                if name == "started":
                    st.session_state.thread_id = data["thread_id"]
                elif name == "node_complete":
                    label = STEPS.get(data["node"], data["node"])
                    st.session_state.log.append(label)
                    st.write(f"✓ {label}")
                elif name == "awaiting_human":
                    st.session_state.awaiting_human = data
                    status.update(label="Draft ready for your review", state="complete", expanded=False)
                elif name == "done":
                    st.session_state.awaiting_human = None
                    st.session_state.final_report = data.get("report_markdown")
                    st.session_state.final_path = data.get("report_path")
                    status.update(label="Research complete", state="complete", expanded=False)
                elif name == "error":
                    st.session_state.error = data["message"]
                    status.update(label="Run failed", state="error")
        except httpx.HTTPError as exc:
            st.session_state.error = f"Could not reach the API at {API_BASE}: {exc}"
            status.update(label="Run failed", state="error")


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    online = api_online()
    if online:
        st.success(f"API connected · {API_BASE}")
    else:
        st.error(f"API offline · {API_BASE}")
        st.caption("Start it with `uvicorn api.server:app --reload`")

    user_id = st.text_input("User ID", value="default_user", help="Used for long-term memory across sessions.")
    auto_approve = st.toggle("Auto-approve draft", value=False, help="Skip the human review step.")

    st.divider()
    st.subheader("Pipeline")
    for label in STEPS.values():
        st.caption(f"• {label}")

    if st.session_state.thread_id:
        st.divider()
        st.caption(f"Session: `{st.session_state.thread_id}`")
    if st.button("New session", use_container_width=True):
        for key in ("thread_id", "awaiting_human", "final_report", "final_path", "error"):
            st.session_state[key] = None
        st.session_state.log = []
        st.rerun()

# --------------------------------------------------------------------------- main
st.title("🔎 Argus")
st.caption(
    "A self-correcting multi-agent research assistant — plans, researches in parallel, "
    "critiques its own drafts, and asks for your approval before finalizing."
)

st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.question = example

question = st.text_area(
    "Research question",
    key="question",
    height=100,
    placeholder="Ask anything that needs real research…",
)

if st.button("Run research", type="primary", disabled=not (question.strip() and online)):
    st.session_state.awaiting_human = None
    st.session_state.final_report = None
    st.session_state.log = []
    run_stream(
        f"{API_BASE}/research",
        {"question": question, "user_id": user_id, "auto_approve": auto_approve},
    )

if st.session_state.error:
    st.error(st.session_state.error)

draft = st.session_state.awaiting_human
if draft:
    st.divider()
    st.subheader("📝 Draft ready for review")
    critique = draft.get("critique")
    if critique:
        c1, c2 = st.columns(2)
        c1.metric("Critic verdict", "Passed" if critique.get("passed") else "Needs work")
        c2.metric("Citation coverage", f"{critique.get('citation_coverage_score', 0):.0%}")
        if critique.get("issues"):
            with st.expander("Critic notes"):
                for issue in critique["issues"]:
                    st.write(f"- {issue}")

    with st.container(border=True):
        st.markdown(draft["draft_report_markdown"])

    with st.form("review"):
        feedback = st.text_area("Requested changes", placeholder="e.g. Add a section on pricing")
        a, r = st.columns(2)
        approve = a.form_submit_button("✅ Approve", type="primary", use_container_width=True)
        revise = r.form_submit_button("✏️ Request changes", use_container_width=True)

    if approve or (revise and feedback.strip()):
        run_stream(
            f"{API_BASE}/research/resume",
            {"thread_id": st.session_state.thread_id, "feedback": "approve" if approve else feedback},
        )
        st.rerun()
    elif revise:
        st.warning("Describe what you'd like changed first.")

if st.session_state.final_report:
    st.divider()
    st.subheader("📄 Final report")
    if st.session_state.final_path:
        st.caption(f"Saved to `{st.session_state.final_path}`")
    st.download_button(
        "Download Markdown",
        st.session_state.final_report,
        file_name=os.path.basename(st.session_state.final_path or "report.md"),
        mime="text/markdown",
    )
    with st.container(border=True):
        st.markdown(st.session_state.final_report)
