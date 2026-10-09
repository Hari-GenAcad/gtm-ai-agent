from __future__ import annotations

import os
from pathlib import Path
import re

import streamlit as st

from gtm_agent.config import load_local_env
from gtm_agent.ingestion import load_bytes, load_public_google_sheet, merge_documents
from gtm_agent.llm import FakeLLM, FakeScenario, GeminiLLM, OpenAILLM
from gtm_agent.models.schemas import CampaignRequest, HumanReviewResponse, WorkflowStatus
from gtm_agent.orchestration import WorkflowRunner


def _display_body(body: str) -> str:
    """Keep citations in exported state while removing inline citation clutter from the UI."""
    return re.sub(r"\s*\[Sources?:[^\]]+\]", "", body, flags=re.IGNORECASE).strip()


def _render_asset(asset) -> None:
    with st.container(border=True):
        if asset.title:
            st.subheader(asset.title)
        st.markdown(_display_body(asset.body))
        if asset.source_ids:
            st.caption("Evidence: " + " · ".join(asset.source_ids))


load_local_env()
st.set_page_config(page_title="GTM Content Agent", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1280px; padding-top: 2.5rem; padding-bottom: 4rem; }
    [data-testid="stMarkdownContainer"] p { line-height: 1.65; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
    div[data-testid="stExpander"] { border-radius: 0.65rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Evidence-grounded GTM Content Agent")
st.caption("Turn approved product evidence into a reviewed, multi-format campaign suite.")

uploaded = st.file_uploader(
    "Approved source documents",
    type=["txt", "md", "pdf", "csv", "xlsx", "zip"],
    accept_multiple_files=True,
)
google_sheet_url = st.text_input("Public Google Sheet URL (optional)", "")
objective = st.text_input("Campaign objective", "Introduce the product with evidence-grounded messaging")
audience = st.text_input("Target audience override (optional)", "")
tone = st.text_input("Tone override (optional)", "")
provider = st.selectbox("Provider", ["Deterministic fake", "Gemini", "OpenAI"])
scenario = st.selectbox("Fake scenario", [item.value for item in FakeScenario])

if st.button("Start run", type="primary", disabled=not uploaded and not google_sheet_url.strip()):
    try:
        documents = [document for item in uploaded for document in load_bytes(item.name, item.getvalue())]
        if google_sheet_url.strip():
            documents.extend(load_public_google_sheet(google_sheet_url))
    except (ValueError, RuntimeError, UnicodeError) as exc:
        st.error(f"Unable to load source: {exc}")
        st.stop()
    brief = merge_documents(documents)
    if provider == "Gemini" and not os.getenv("GEMINI_API_KEY"):
        st.error("Set GEMINI_API_KEY before starting a Gemini run.")
        st.stop()
    if provider == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
        st.error("Set OPENAI_API_KEY before starting an OpenAI run.")
        st.stop()
    if provider == "Gemini":
        llm = GeminiLLM()
    elif provider == "OpenAI":
        llm = OpenAILLM()
    else:
        llm = FakeLLM(FakeScenario(scenario))
    previous_runner = st.session_state.get("runner")
    if previous_runner:
        previous_runner.close()
    runner = WorkflowRunner(llm, checkpoint_path=Path(".gtm-checkpoints.sqlite"))
    thread_id, state = runner.start(
        brief,
        CampaignRequest(objective=objective, audience=audience, tone=tone),
        documents=documents,
    )
    st.session_state.runner = runner
    st.session_state.thread_id = thread_id
    st.session_state.workflow_state = state
    st.session_state.checkpoint_thread = thread_id

state = st.session_state.get("workflow_state")
if state:
    status_value = getattr(state["status"], "value", state["status"])
    status_col, revision_col, retrieval_col = st.columns(3)
    status_col.metric("Workflow status", str(status_value).replace("_", " ").title())
    revision_col.metric("Revisions", state.get("revision_count", 0))
    retrieval_col.metric(
        "Retrieval",
        str(state.get("metadata", {}).get("retrieval_mode", "not run")).replace("_", " ").title(),
    )
    st.caption(f"Checkpoint thread: {st.session_state.get('checkpoint_thread', 'unknown')}")

    if state["status"] == WorkflowStatus.WAITING_HUMAN:
        st.warning(state["human_request"].reason)
        comment = st.text_area("Human response context")
        col1, col2, col3 = st.columns(3)
        decision = None
        if col1.button("Approve flagged draft"):
            decision = "approve"
        if col2.button("Reject"):
            decision = "reject"
        if col3.button("Clarify and retry"):
            decision = "clarify"
        if decision:
            if decision != "approve" and not comment.strip():
                st.error("Rejection and clarification require context.")
            else:
                response = HumanReviewResponse(decision=decision, comment=comment or "Explicit UI approval.")
                state = st.session_state.runner.resume(st.session_state.thread_id, response)
                st.session_state.workflow_state = state
                st.rerun()

    evidence_tab, analysis_tab, content_tab, review_tab, history_tab = st.tabs(
        ["Evidence", "Campaign analysis", "Content", "Review", "Action history"]
    )
    with evidence_tab:
        evidence = state.get("evidence", [])
        st.markdown(f"### Retrieved evidence ({len(evidence)})")
        st.caption("Expand a passage to inspect the exact approved context used by the agent.")
        for item in evidence:
            label = f"{item.source_id} · passage {item.chunk_index + 1} · relevance {item.score:.3f}"
            with st.expander(label):
                st.markdown(item.text)
                st.caption(f"Passage ID: {item.passage_id}")

    with analysis_tab:
        analysis = state.get("analysis")
        if analysis:
            left, right = st.columns(2)
            with left:
                st.markdown("#### Target audience")
                for item in analysis.target_audience:
                    st.markdown(f"- {item}")
            with right:
                st.markdown("#### Tone")
                st.markdown(analysis.tone)
            st.markdown("#### Value proposition")
            st.info(analysis.value_proposition)
            st.markdown("#### Key messages")
            for message in analysis.key_messages:
                st.markdown(f"- {message}")
            if analysis.missing_facts:
                st.markdown("#### Missing facts")
                for fact in analysis.missing_facts:
                    st.warning(fact)
            with st.expander("Why the agent chose this positioning"):
                st.write(analysis.rationale)

    with content_tab:
        suite = state.get("content")
        if suite:
            linkedin_tab, email_tab, blog_tab, ads_tab = st.tabs(["LinkedIn", "Email", "Blog", "Ads"])
            with linkedin_tab:
                _render_asset(suite.linkedin)
            with email_tab:
                _render_asset(suite.email)
            with blog_tab:
                _render_asset(suite.blog)
            with ads_tab:
                ad_columns = st.columns(min(3, len(suite.ads)))
                for index, asset in enumerate(suite.ads):
                    with ad_columns[index % len(ad_columns)]:
                        _render_asset(asset)
            if suite.missing_facts:
                with st.expander("Missing facts to resolve"):
                    for fact in suite.missing_facts:
                        st.markdown(f"- {fact}")
            st.download_button(
                "Export content JSON",
                suite.model_dump_json(indent=2),
                file_name="gtm-content-suite.json",
                mime="application/json",
            )

    with review_tab:
        review = state.get("review")
        if review:
            if review.approved:
                st.success(review.summary)
            else:
                st.error(review.summary)
            if review.findings:
                st.markdown("#### Findings")
                for finding in review.findings:
                    with st.container(border=True):
                        st.markdown(f"**{finding.category.title()} · {finding.severity.title()}**")
                        st.write(finding.message)
                        st.caption("Recommended action: " + finding.action)
            with st.expander("Deterministic checks"):
                for name, passed in review.deterministic_checks.items():
                    icon = "✅" if passed else "❌"
                    st.write(f"{icon} {name.replace('_', ' ').title()}")

    with history_tab:
        history = state.get("history", [])
        st.dataframe(
            [
                {
                    "Step": item.step,
                    "Action": item.action.replace("_", " ").title(),
                    "Outcome": item.outcome.title(),
                    "Detail": item.detail,
                }
                for item in history
            ],
            width="stretch",
            hide_index=True,
        )
