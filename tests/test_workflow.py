import socket

import pytest

from gtm_agent.llm.fake import FakeLLM, FakeScenario
from gtm_agent.llm.base import ProviderFailure
from gtm_agent.models.schemas import (
    CampaignRequest,
    ErrorKind,
    HumanReviewResponse,
    ProductBrief,
    WorkflowStage,
    WorkflowStatus,
)
from gtm_agent.orchestration import WorkflowRunner


def run(scenario, brief, campaign, **kwargs):
    runner = WorkflowRunner(FakeLLM(scenario))
    thread_id, state = runner.start(brief, campaign, **kwargs)
    return runner, thread_id, state


def test_happy_path_generates_all_formats(brief, campaign) -> None:
    _, _, state = run(FakeScenario.HAPPY, brief, campaign)
    assert state["status"] == WorkflowStatus.APPROVED
    assert state["stage"] == WorkflowStage.COMPLETE
    assert len(state["content"].ads) == 3
    assert state["review"].approved
    assert state["analysis"].target_audience
    assert [item.action for item in state["history"]] == [
        "initialize", "supervisor", "retrieve", "supervisor", "analyze", "supervisor", "generate",
        "supervisor", "review", "supervisor", "complete",
    ]


def test_failed_review_routes_to_revision_then_passes(brief, campaign) -> None:
    _, _, state = run(FakeScenario.REVIEW_FAIL_THEN_PASS, brief, campaign)
    assert state["status"] == WorkflowStatus.APPROVED
    assert state["revision_count"] == 1
    assert len(state["review_history"]) == 2
    assert "revise" in [item.action for item in state["history"]]


def test_revision_limit_pauses_and_rejection_stays_unapproved(brief, campaign) -> None:
    runner, thread_id, state = run(FakeScenario.REVIEW_ALWAYS_FAILS, brief, campaign, max_revisions=2)
    assert state["status"] == WorkflowStatus.WAITING_HUMAN
    assert state["revision_count"] == 2
    assert not state["review"].approved
    state = runner.resume(thread_id, HumanReviewResponse(decision="reject", comment="unsafe claims"))
    assert state["status"] == WorkflowStatus.UNAPPROVED
    assert state["stage"] == WorkflowStage.FAILED


def test_human_approval_is_not_mislabeled_automated_approval(brief, campaign) -> None:
    runner, thread_id, state = run(FakeScenario.REVIEW_ALWAYS_FAILS, brief, campaign, max_revisions=0)
    history_before = len(state["history"])
    state = runner.resume(thread_id, HumanReviewResponse(decision="approve"))
    assert state["status"] == WorkflowStatus.UNAPPROVED
    assert state["stage"] == WorkflowStage.COMPLETE
    assert len(state["history"]) > history_before
    assert state["metadata"]["human_override"] is True


def test_human_clarification_resumes_from_retrieval(campaign) -> None:
    brief = ProductBrief(title="Opaque", source_id="brief.txt", content="Unrelated approved text.")
    runner, thread_id, state = run(FakeScenario.HAPPY, brief, campaign)
    assert state["status"] == WorkflowStatus.WAITING_HUMAN
    state = runner.resume(
        thread_id,
        HumanReviewResponse(
            decision="clarify",
            comment="LaunchPad helps revenue teams run a campaign.",
        ),
    )
    assert state["status"] == WorkflowStatus.APPROVED
    assert sum(item.action == "retrieve" for item in state["history"]) == 2


@pytest.mark.parametrize(
    ("scenario", "kind"),
    [
        (FakeScenario.PROVIDER_FAILURE, ErrorKind.PROVIDER),
        (FakeScenario.MALFORMED_OUTPUT, ErrorKind.MALFORMED_OUTPUT),
    ],
)
def test_model_failures_retry_then_stop_safely(scenario, kind, brief, campaign) -> None:
    _, _, state = run(scenario, brief, campaign)
    matching = [error for error in state["errors"] if error.kind == kind]
    assert len(matching) == 2
    assert matching[0].recoverable is True
    assert matching[-1].recoverable is False
    assert state["status"] == WorkflowStatus.ERROR


def test_invalid_supervisor_action_is_rejected_then_recovered(brief, campaign) -> None:
    _, _, state = run(FakeScenario.INVALID_SUPERVISOR_ACTION, brief, campaign)
    assert state["status"] == WorkflowStatus.APPROVED
    assert any(error.kind == ErrorKind.MALFORMED_OUTPUT for error in state["errors"])


def test_missing_evidence_interrupts_and_preserves_state(brief, campaign) -> None:
    runner, thread_id, state = run(FakeScenario.MISSING_EVIDENCE, brief, campaign)
    assert state["status"] == WorkflowStatus.WAITING_HUMAN
    assert state["brief"] == brief
    assert runner.get_state(thread_id)["human_request"] == state["human_request"]


def test_contradictory_evidence_requests_human(campaign) -> None:
    brief = ProductBrief(
        title="LaunchPad",
        source_id="brief.md",
        content="LaunchPad for revenue teams.\nLaunch date: May 1\nLaunch date: June 1",
    )
    _, _, state = run(FakeScenario.HAPPY, brief, campaign)
    assert state["status"] == WorkflowStatus.WAITING_HUMAN
    assert "contradictory" in state["human_request"].reason.lower()


def test_fake_path_does_not_use_network(monkeypatch, brief, campaign) -> None:
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", blocked)
    _, _, state = run(FakeScenario.HAPPY, brief, campaign)
    assert state["status"] == WorkflowStatus.APPROVED


def test_workflow_terminates_within_step_limit(brief, campaign) -> None:
    _, _, state = run(FakeScenario.REVIEW_FAIL_THEN_PASS, brief, campaign, max_steps=12)
    assert state["step_count"] <= 12
    assert state["stage"] in {WorkflowStage.COMPLETE, WorkflowStage.FAILED}


def test_actual_step_limit_exhaustion_fails_safely(brief, campaign) -> None:
    _, _, state = run(FakeScenario.HAPPY, brief, campaign, max_steps=1)
    assert state["status"] == WorkflowStatus.ERROR
    assert state["stage"] == WorkflowStage.FAILED
    assert any(error.kind == ErrorKind.LIMIT for error in state["errors"])


def test_supervisor_provider_failure_is_bounded(brief, campaign) -> None:
    class FailingSupervisor(FakeLLM):
        def decide(self, state_summary):
            raise ProviderFailure("supervisor unavailable")

    runner = WorkflowRunner(FailingSupervisor())
    _, state = runner.start(brief, campaign)
    assert state["status"] == WorkflowStatus.ERROR
    assert len([error for error in state["errors"] if error.kind == ErrorKind.PROVIDER]) == 2
    assert not state["evidence"]


def test_review_provider_failure_is_bounded(brief, campaign) -> None:
    class FailingReviewer(FakeLLM):
        def review(self, *args, **kwargs):
            raise ProviderFailure("review unavailable")

    runner = WorkflowRunner(FailingReviewer())
    _, state = runner.start(brief, campaign)
    assert state["status"] == WorkflowStatus.ERROR
    assert state["content"] is not None
    assert state["review"] is None
    assert len([error for error in state["errors"] if error.kind == ErrorKind.PROVIDER]) == 2


def test_revision_provider_failure_does_not_consume_revision(brief, campaign) -> None:
    class FailingRevision(FakeLLM):
        def __init__(self):
            super().__init__(FakeScenario.REVIEW_FAIL_THEN_PASS)

        def generate(self, brief, campaign, evidence, revision_findings=None, analysis=None):
            if revision_findings:
                raise ProviderFailure("revision unavailable")
            return super().generate(brief, campaign, evidence, revision_findings, analysis)

    runner = WorkflowRunner(FailingRevision())
    _, state = runner.start(brief, campaign)
    assert state["status"] == WorkflowStatus.ERROR
    assert state["revision_count"] == 0
    assert state["review"] is not None and not state["review"].approved


def test_human_factual_update_is_retained_and_retrieved(campaign) -> None:
    brief = ProductBrief(title="Opaque", source_id="brief.txt", content="Unrelated approved text.")
    runner, thread_id, state = run(FakeScenario.HAPPY, brief, campaign)
    state = runner.resume(
        thread_id,
        HumanReviewResponse(
            decision="clarify",
            comment="Use this approved correction.",
            factual_updates={"Product": "LaunchPad helps revenue teams launch campaigns"},
        ),
    )
    assert state["status"] == WorkflowStatus.APPROVED
    assert "Product: LaunchPad helps revenue teams" in state["brief"].content
    assert any("LaunchPad helps revenue teams" in item.text for item in state["evidence"])


def test_risky_claim_pauses_before_any_generation(campaign) -> None:
    brief = ProductBrief(
        title="LaunchPad",
        source_id="brief.md",
        content="LaunchPad is the world's best campaign product for revenue teams.",
    )
    _, _, state = run(FakeScenario.HAPPY, brief, campaign)
    assert state["status"] == WorkflowStatus.WAITING_HUMAN
    assert state["content"] is None
    assert any("unsupported" in issue.lower() for issue in state["human_request"].unresolved_findings)


def test_campaign_analysis_infers_missing_audience_and_tone(brief) -> None:
    request = CampaignRequest(objective="Prepare a LaunchPad launch campaign")
    _, _, state = run(FakeScenario.HAPPY, brief, request)
    assert state["status"] == WorkflowStatus.APPROVED
    assert "revenue leaders" in state["analysis"].target_audience
    assert state["campaign"].audience == "revenue leaders"
    assert state["campaign"].tone == "clear, confident, and practical"
    assert state["analysis"].value_proposition
    assert "analyze" in [item.action for item in state["history"]]


def test_campaign_analysis_preserves_user_overrides(brief) -> None:
    request = CampaignRequest(
        objective="Prepare a campaign",
        audience="technical founders",
        tone="direct and analytical",
    )
    _, _, state = run(FakeScenario.HAPPY, brief, request)
    assert state["campaign"].audience == "technical founders"
    assert state["campaign"].tone == "direct and analytical"
    assert state["analysis"].target_audience == ["technical founders"]


def test_campaign_analysis_provider_failure_is_bounded(brief, campaign) -> None:
    class FailingAnalysis(FakeLLM):
        def analyze(self, *args, **kwargs):
            raise ProviderFailure("analysis unavailable")

    runner = WorkflowRunner(FailingAnalysis())
    _, state = runner.start(brief, campaign)
    assert state["status"] == WorkflowStatus.ERROR
    assert state["analysis"] is None
    assert len([error for error in state["errors"] if error.kind == ErrorKind.PROVIDER]) == 2


def test_multi_document_flow_retrieves_attributed_context(campaign) -> None:
    product = ProductBrief(
        title="LaunchPad",
        source_id="product.md",
        content="LaunchPad helps revenue teams coordinate campaign messaging.",
    )
    calendar = ProductBrief(
        title="Launch calendar",
        source_id="calendar.csv",
        content="LaunchPad launch date is November 1. Pricing is INR 4,999.",
        media_type="spreadsheet",
    )
    runner = WorkflowRunner(FakeLLM(), vector_persist_directory=None)
    _, state = runner.start(product, campaign, documents=[product, calendar])
    assert state["status"] == WorkflowStatus.APPROVED
    assert state["metadata"]["retrieval_mode"] == "vector"
    assert {item.source_id for item in state["evidence"]} == {"product.md", "calendar.csv"}
    assert set(state["content"].linkedin.source_ids) == {"product.md", "calendar.csv"}


def test_sqlite_checkpoint_resumes_after_runner_restart(tmp_path, brief, campaign) -> None:
    database = tmp_path / "workflow.sqlite"
    first_runner = WorkflowRunner(
        FakeLLM(FakeScenario.REVIEW_ALWAYS_FAILS),
        checkpoint_path=database,
        vector_persist_directory=None,
    )
    thread_id, paused = first_runner.start(brief, campaign, max_revisions=0)
    assert paused["status"] == WorkflowStatus.WAITING_HUMAN
    first_runner.close()

    restarted_runner = WorkflowRunner(
        FakeLLM(FakeScenario.HAPPY),
        checkpoint_path=database,
        vector_persist_directory=None,
    )
    assert restarted_runner.get_state(thread_id)["status"] == WorkflowStatus.WAITING_HUMAN
    completed = restarted_runner.resume(thread_id, HumanReviewResponse(decision="approve"))
    restarted_runner.close()

    assert completed["status"] == WorkflowStatus.UNAPPROVED
    assert completed["stage"] == WorkflowStage.COMPLETE
    assert completed["metadata"]["human_override"] is True


def test_runner_rejects_two_checkpoint_backends(tmp_path) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    with pytest.raises(ValueError, match="either checkpointer or checkpoint_path"):
        WorkflowRunner(
            FakeLLM(),
            checkpointer=InMemorySaver(),
            checkpoint_path=tmp_path / "workflow.sqlite",
        )


def test_persistently_illegal_supervisor_decisions_use_safe_routing(brief, campaign) -> None:
    class IllegalSupervisor(FakeLLM):
        def __init__(self):
            super().__init__(FakeScenario.REVIEW_ALWAYS_FAILS)

        def decide(self, state_summary):
            return {"action": "revise", "reason": "always invalid here"}

    runner = WorkflowRunner(IllegalSupervisor(), vector_persist_directory=None)
    _, state = runner.start(brief, campaign, max_revisions=0)

    assert state["status"] == WorkflowStatus.WAITING_HUMAN
    assert any("fallback" in item.detail.lower() for item in state["history"])
    assert not state.get("metadata", {}).get("fatal_error")
