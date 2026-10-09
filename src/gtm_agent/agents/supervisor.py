from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from gtm_agent.llm.base import MalformedModelOutput
from gtm_agent.models.schemas import SupervisorActionName, SupervisorDecision, WorkflowState


def state_summary(state: WorkflowState) -> dict[str, Any]:
    metadata = state.get("metadata", {})
    review = state.get("review")
    return {
        "retrieval_attempted": bool(metadata.get("retrieval_attempted")),
        "has_evidence": bool(state.get("evidence")),
        "has_analysis": state.get("analysis") is not None,
        "evidence_issues": bool(metadata.get("evidence_issues"))
        and not bool(metadata.get("resolved_evidence_issues")),
        "has_content": state.get("content") is not None,
        "has_review": review is not None,
        "review_approved": bool(review and review.approved),
        "revision_count": state.get("revision_count", 0),
        "max_revisions": state.get("max_revisions", 2),
        "fatal_error": bool(metadata.get("fatal_error")),
        "human_override": bool(metadata.get("human_override")),
        "human_rejected": bool(metadata.get("human_rejected")),
        "step_count": state.get("step_count", 0),
        "max_steps": state.get("max_steps", 20),
    }


def legal_actions(state: WorkflowState) -> set[SupervisorActionName]:
    summary = state_summary(state)
    if summary["fatal_error"] or summary["human_rejected"]:
        return {SupervisorActionName.FAIL}
    if summary["human_override"]:
        return {SupervisorActionName.COMPLETE}
    if not summary["retrieval_attempted"]:
        return {SupervisorActionName.RETRIEVE, SupervisorActionName.FAIL}
    if not summary["has_evidence"]:
        return {SupervisorActionName.REQUEST_HUMAN, SupervisorActionName.FAIL}
    if summary["evidence_issues"]:
        return {SupervisorActionName.REQUEST_HUMAN, SupervisorActionName.FAIL}
    if not summary["has_analysis"]:
        return {SupervisorActionName.ANALYZE, SupervisorActionName.FAIL}
    if not summary["has_content"]:
        return {SupervisorActionName.GENERATE, SupervisorActionName.FAIL}
    if not summary["has_review"]:
        return {SupervisorActionName.REVIEW, SupervisorActionName.FAIL}
    if summary["review_approved"]:
        return {SupervisorActionName.COMPLETE, SupervisorActionName.FAIL}
    if summary["revision_count"] < summary["max_revisions"]:
        return {SupervisorActionName.REVISE, SupervisorActionName.REQUEST_HUMAN, SupervisorActionName.FAIL}
    return {SupervisorActionName.REQUEST_HUMAN, SupervisorActionName.FAIL}


def validate_decision(raw: Any, state: WorkflowState) -> SupervisorDecision:
    try:
        decision = SupervisorDecision.model_validate(raw)
    except ValidationError as exc:
        raise MalformedModelOutput(f"invalid supervisor output: {exc}") from exc
    allowed = legal_actions(state)
    if decision.action not in allowed:
        names = ", ".join(sorted(item.value for item in allowed))
        raise MalformedModelOutput(
            f"action {decision.action.value!r} is invalid for current state; allowed: {names}"
        )
    return decision
