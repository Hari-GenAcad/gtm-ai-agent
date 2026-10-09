from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from gtm_agent.agents.reviewer import review_content
from gtm_agent.agents.supervisor import legal_actions, state_summary, validate_decision
from gtm_agent.llm.base import LLMClient, MalformedModelOutput, ProviderFailure
from gtm_agent.llm.fake import FakeScenario
from gtm_agent.models.schemas import (
    ActionRecord,
    CampaignAnalysis,
    CampaignRequest,
    ErrorKind,
    HumanReviewRequest,
    HumanReviewResponse,
    ProductBrief,
    SupervisorActionName,
    SupervisorDecision,
    WorkflowError,
    WorkflowStage,
    WorkflowState,
    WorkflowStatus,
)
from gtm_agent.retrieval.local import LexicalRetriever, analyze_evidence_quality
from gtm_agent.retrieval.vector import VectorRetriever


def _serializer() -> JsonPlusSerializer:
    from gtm_agent.models import schemas

    allowed = [
        value
        for value in vars(schemas).values()
        if isinstance(value, type) and value.__module__ == schemas.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def _memory_saver() -> InMemorySaver:
    return InMemorySaver(serde=_serializer())


def _sqlite_saver(path: str | Path) -> tuple[SqliteSaver, sqlite3.Connection]:
    database = Path(path).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, check_same_thread=False)
    return SqliteSaver(connection, serde=_serializer()), connection


def _history(state: WorkflowState, action: str, outcome: str, detail: str) -> list[ActionRecord]:
    return [
        *state.get("history", []),
        ActionRecord(
            step=state.get("step_count", 0),
            action=action,
            outcome=outcome,
            detail=detail,
        ),
    ]


def _errors(
    state: WorkflowState,
    kind: ErrorKind,
    message: str,
    *,
    recoverable: bool,
    attempt: int,
) -> list[WorkflowError]:
    return [
        *state.get("errors", []),
        WorkflowError(kind=kind, message=message, recoverable=recoverable, attempt=attempt),
    ]


def _failure_kind(exc: Exception) -> ErrorKind:
    if isinstance(exc, ProviderFailure):
        return ErrorKind.PROVIDER
    if isinstance(exc, MalformedModelOutput):
        return ErrorKind.MALFORMED_OUTPUT
    return ErrorKind.INTERNAL


def build_graph(
    llm: LLMClient,
    *,
    checkpointer: Any | None = None,
    vector_persist_directory: str | None = ".gtm-vector-store",
):
    saver = checkpointer or _memory_saver()

    def initialize(state: WorkflowState) -> dict[str, Any]:
        return {
            "stage": WorkflowStage.SUPERVISE,
            "status": WorkflowStatus.RUNNING,
            "evidence": state.get("evidence", []),
            "documents": state.get("documents", [state["brief"]]),
            "analysis": state.get("analysis"),
            "content": state.get("content"),
            "review": state.get("review"),
            "review_history": state.get("review_history", []),
            "revision_count": state.get("revision_count", 0),
            "max_revisions": state.get("max_revisions", 2),
            "step_count": state.get("step_count", 0),
            "max_steps": state.get("max_steps", 20),
            "provider_attempts": 0,
            "max_provider_attempts": state.get("max_provider_attempts", 2),
            "decision": None,
            "history": _history(state, "initialize", "success", "Workflow state initialized."),
            "errors": state.get("errors", []),
            "human_request": None,
            "human_response": None,
            "metadata": state.get("metadata", {}),
        }

    def supervisor(state: WorkflowState) -> dict[str, Any]:
        next_step = state.get("step_count", 0) + 1
        if next_step > state.get("max_steps", 20):
            decision = SupervisorDecision(action=SupervisorActionName.FAIL, reason="Maximum graph steps reached.")
            return {
                "stage": WorkflowStage.SUPERVISE,
                "step_count": next_step,
                "decision": decision,
                "errors": _errors(
                    state, ErrorKind.LIMIT, "Maximum graph steps reached.", recoverable=False, attempt=1
                ),
                "history": _history(state, "supervisor", "error", decision.reason),
            }

        errors = list(state.get("errors", []))
        max_attempts = state.get("max_provider_attempts", 2)
        attempt_kinds: list[ErrorKind] = []
        for attempt in range(1, max_attempts + 1):
            try:
                decision = validate_decision(llm.decide(state_summary(state)), state)
                return {
                    "stage": WorkflowStage.SUPERVISE,
                    "step_count": next_step,
                    "decision": decision,
                    "provider_attempts": 0,
                    "errors": errors,
                    "history": _history(state, "supervisor", "selected", f"{decision.action}: {decision.reason}"),
                }
            except (ProviderFailure, MalformedModelOutput) as exc:
                attempt_kinds.append(_failure_kind(exc))
                errors.append(
                    WorkflowError(
                        kind=_failure_kind(exc),
                        message=str(exc),
                        recoverable=attempt < max_attempts,
                        attempt=attempt,
                    )
                )

        if attempt_kinds and all(kind == ErrorKind.MALFORMED_OUTPUT for kind in attempt_kinds):
            allowed = legal_actions(state)
            preference = [
                SupervisorActionName.RETRIEVE,
                SupervisorActionName.ANALYZE,
                SupervisorActionName.GENERATE,
                SupervisorActionName.REVIEW,
                SupervisorActionName.REVISE,
                SupervisorActionName.COMPLETE,
                SupervisorActionName.REQUEST_HUMAN,
                SupervisorActionName.FAIL,
            ]
            fallback = next(action for action in preference if action in allowed)
            decision = SupervisorDecision(
                action=fallback,
                reason="Deterministic fallback selected after invalid supervisor outputs.",
            )
            return {
                "stage": WorkflowStage.SUPERVISE,
                "step_count": next_step,
                "decision": decision,
                "provider_attempts": max_attempts,
                "errors": errors,
                "history": _history(state, "supervisor", "selected", decision.reason),
            }

        metadata = {**state.get("metadata", {}), "fatal_error": True}
        decision = SupervisorDecision(action=SupervisorActionName.FAIL, reason="Supervisor retries exhausted.")
        return {
            "stage": WorkflowStage.SUPERVISE,
            "step_count": next_step,
            "decision": decision,
            "provider_attempts": max_attempts,
            "errors": errors,
            "metadata": metadata,
            "history": _history(state, "supervisor", "error", decision.reason),
        }

    def retrieve(state: WorkflowState) -> dict[str, Any]:
        campaign = state["campaign"]
        query = f"{state['brief'].title} {campaign.objective} {campaign.audience}"
        forced_empty = getattr(llm, "scenario", None) == FakeScenario.MISSING_EVIDENCE
        retrieval_mode = "vector"
        retrieval_error = None
        if forced_empty:
            evidence = []
        else:
            try:
                retriever = VectorRetriever(
                    state.get("documents", [state["brief"]]),
                    persist_directory=vector_persist_directory,
                )
                evidence = retriever.search(query, limit=5)
                retrieval_mode = retriever.mode
                retrieval_error = retriever.error
            except Exception as exc:
                retrieval_mode = "lexical_fallback"
                retrieval_error = str(exc)
                evidence = [
                    item
                    for document in state.get("documents", [state["brief"]])
                    for item in LexicalRetriever(document).search(query, limit=5)
                ][:5]
        source_text = "\n".join(item.content for item in state.get("documents", [state["brief"]]))
        evidence_issues = analyze_evidence_quality(source_text)
        metadata = {
            **state.get("metadata", {}),
            "retrieval_attempted": True,
            "evidence_issues": evidence_issues,
            "retrieval_mode": retrieval_mode,
            "retrieval_error": retrieval_error,
        }
        update: dict[str, Any] = {
            "stage": WorkflowStage.RETRIEVE,
            "evidence": evidence,
            "metadata": metadata,
            "history": _history(
                state,
                "retrieve",
                "success" if evidence else "error",
                f"Retrieved {len(evidence)} attributed passage(s).",
            ),
        }
        if not evidence:
            update["errors"] = _errors(
                state,
                ErrorKind.RETRIEVAL,
                "No relevant evidence was retrieved from the approved brief.",
                recoverable=True,
                attempt=1,
            )
        return update

    def analyze(state: WorkflowState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        max_attempts = state.get("max_provider_attempts", 2)
        for attempt in range(1, max_attempts + 1):
            try:
                result: CampaignAnalysis = llm.analyze(
                    state["brief"], state["campaign"], state["evidence"]
                )
                campaign = state["campaign"].model_copy(
                    update={
                        "audience": state["campaign"].audience or ", ".join(result.target_audience),
                        "tone": state["campaign"].tone or result.tone,
                    }
                )
                return {
                    "stage": WorkflowStage.ANALYZE,
                    "analysis": result,
                    "campaign": campaign,
                    "provider_attempts": 0,
                    "errors": errors,
                    "history": _history(
                        state,
                        "analyze",
                        "success",
                        f"Selected audience '{campaign.audience}' and tone '{campaign.tone}'.",
                    ),
                }
            except (ProviderFailure, MalformedModelOutput) as exc:
                errors.append(
                    WorkflowError(
                        kind=_failure_kind(exc),
                        message=str(exc),
                        recoverable=attempt < max_attempts,
                        attempt=attempt,
                    )
                )
        return {
            "stage": WorkflowStage.ANALYZE,
            "provider_attempts": max_attempts,
            "errors": errors,
            "metadata": {**state.get("metadata", {}), "fatal_error": True},
            "history": _history(state, "analyze", "error", "Campaign-analysis retries exhausted."),
        }

    def call_generation(state: WorkflowState, *, revision: bool) -> dict[str, Any]:
        action = "revise" if revision else "generate"
        findings = state["review"].findings if revision and state.get("review") else None
        errors = list(state.get("errors", []))
        max_attempts = state.get("max_provider_attempts", 2)
        for attempt in range(1, max_attempts + 1):
            try:
                content = llm.generate(
                    state["brief"],
                    state["campaign"],
                    state["evidence"],
                    findings,
                    state.get("analysis"),
                )
                update: dict[str, Any] = {
                    "stage": WorkflowStage.REVISE if revision else WorkflowStage.GENERATE,
                    "content": content,
                    "review": None,
                    "provider_attempts": 0,
                    "errors": errors,
                    "history": _history(state, action, "success", "Generated all required content formats."),
                }
                if revision:
                    update["revision_count"] = state.get("revision_count", 0) + 1
                return update
            except (ProviderFailure, MalformedModelOutput) as exc:
                errors.append(
                    WorkflowError(
                        kind=_failure_kind(exc),
                        message=str(exc),
                        recoverable=attempt < max_attempts,
                        attempt=attempt,
                    )
                )
        return {
            "stage": WorkflowStage.REVISE if revision else WorkflowStage.GENERATE,
            "provider_attempts": max_attempts,
            "errors": errors,
            "metadata": {**state.get("metadata", {}), "fatal_error": True},
            "history": _history(state, action, "error", "Model retries exhausted."),
        }

    def generate(state: WorkflowState) -> dict[str, Any]:
        return call_generation(state, revision=False)

    def revise(state: WorkflowState) -> dict[str, Any]:
        return call_generation(state, revision=True)

    def review(state: WorkflowState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        max_attempts = state.get("max_provider_attempts", 2)
        for attempt in range(1, max_attempts + 1):
            try:
                result = review_content(
                    llm,
                    state["brief"],
                    state["campaign"],
                    state["evidence"],
                    state["content"],
                    state.get("analysis"),
                )
                return {
                    "stage": WorkflowStage.REVIEW,
                    "review": result,
                    "review_history": [*state.get("review_history", []), result],
                    "provider_attempts": 0,
                    "errors": errors,
                    "history": _history(
                        state,
                        "review",
                        "success",
                        f"Review decision: {result.decision}; {len(result.findings)} finding(s).",
                    ),
                }
            except (ProviderFailure, MalformedModelOutput) as exc:
                errors.append(
                    WorkflowError(
                        kind=_failure_kind(exc),
                        message=str(exc),
                        recoverable=attempt < max_attempts,
                        attempt=attempt,
                    )
                )
        return {
            "stage": WorkflowStage.REVIEW,
            "provider_attempts": max_attempts,
            "errors": errors,
            "metadata": {**state.get("metadata", {}), "fatal_error": True},
            "history": _history(state, "review", "error", "Model retries exhausted."),
        }

    def prepare_human(state: WorkflowState) -> dict[str, Any]:
        evidence_issues = state.get("metadata", {}).get("evidence_issues", [])
        if evidence_issues and not state.get("metadata", {}).get("resolved_evidence_issues"):
            request = HumanReviewRequest(
                reason="The approved source contains contradictory or risky claims.",
                questions=["Resolve the listed evidence issue, approve the risk, or reject this run."],
                unresolved_findings=evidence_issues,
            )
        elif not state.get("evidence"):
            request = HumanReviewRequest(
                reason="The approved source did not yield evidence for the campaign request.",
                questions=["Provide factual clarification or reject this run."],
                unresolved_findings=["No attributable evidence was retrieved."],
            )
        else:
            findings = [item.message for item in (state.get("review").findings if state.get("review") else [])]
            request = HumanReviewRequest(
                reason="Automated review remains unresolved after bounded revisions.",
                questions=["Approve the flagged draft, reject it, or provide factual clarification."],
                unresolved_findings=findings or ["The latest automated review did not approve the suite."],
            )
        return {
            "stage": WorkflowStage.HUMAN_REVIEW,
            "status": WorkflowStatus.WAITING_HUMAN,
            "human_request": request,
            "history": _history(state, "request_human", "paused", request.reason),
        }

    def human_interrupt(state: WorkflowState) -> dict[str, Any]:
        raw = interrupt(state["human_request"].model_dump(mode="json"))
        response = HumanReviewResponse.model_validate(raw)
        metadata = dict(state.get("metadata", {}))
        update: dict[str, Any] = {
            "stage": WorkflowStage.HUMAN_REVIEW,
            "status": WorkflowStatus.RUNNING,
            "human_response": response,
            "history": _history(
                state, "human_response", "success", f"Human decision received: {response.decision}."
            ),
        }
        if response.decision == "approve":
            metadata["human_override"] = True
        elif response.decision == "reject":
            metadata["human_rejected"] = True
        else:
            additions = "\n\n".join(f"{key}: {value}" for key, value in response.factual_updates.items())
            if response.comment:
                additions = f"{additions}\n\nClarification: {response.comment}".strip()
            update.update(
                {
                    "brief": state["brief"].model_copy(
                        update={"content": f"{state['brief'].content}\n\n{additions}".strip()}
                    ),
                    "documents": [
                        state["brief"].model_copy(
                            update={"content": f"{state['brief'].content}\n\n{additions}".strip()}
                        ),
                        *state.get("documents", [state["brief"]])[1:],
                    ],
                    "evidence": [],
                    "analysis": None,
                    "content": None,
                    "review": None,
                    "revision_count": 0,
                }
            )
            metadata["retrieval_attempted"] = False
            metadata["resolved_evidence_issues"] = True
        update["metadata"] = metadata
        return update

    def complete(state: WorkflowState) -> dict[str, Any]:
        automated_approval = bool(state.get("review") and state["review"].approved)
        status = WorkflowStatus.APPROVED if automated_approval else WorkflowStatus.UNAPPROVED
        detail = "Automated review approved the content suite."
        if not automated_approval:
            detail = "Human accepted a suite that remains unapproved by automated review."
        return {
            "stage": WorkflowStage.COMPLETE,
            "status": status,
            "history": _history(state, "complete", "success", detail),
        }

    def fail(state: WorkflowState) -> dict[str, Any]:
        rejected = bool(state.get("metadata", {}).get("human_rejected"))
        return {
            "stage": WorkflowStage.FAILED,
            "status": WorkflowStatus.UNAPPROVED if rejected else WorkflowStatus.ERROR,
            "history": _history(
                state,
                "fail",
                "rejected" if rejected else "error",
                "Workflow stopped after explicit human rejection." if rejected else "Workflow stopped safely.",
            ),
        }

    builder = StateGraph(WorkflowState)
    nodes: dict[str, Callable[[WorkflowState], dict[str, Any]]] = {
        "initialize": initialize,
        "supervisor": supervisor,
        "retrieve": retrieve,
        "analyze": analyze,
        "generate": generate,
        "review": review,
        "revise": revise,
        "prepare_human": prepare_human,
        "human_interrupt": human_interrupt,
        "complete": complete,
        "fail": fail,
    }
    for name, function in nodes.items():
        builder.add_node(name, function)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["decision"].action.value,
        {
            "retrieve": "retrieve",
            "analyze": "analyze",
            "generate": "generate",
            "review": "review",
            "revise": "revise",
            "request_human": "prepare_human",
            "complete": "complete",
            "fail": "fail",
        },
    )
    for node in ("retrieve", "analyze", "generate", "review", "revise", "human_interrupt"):
        builder.add_edge(node, "supervisor")
    builder.add_edge("prepare_human", "human_interrupt")
    builder.add_edge("complete", END)
    builder.add_edge("fail", END)
    return builder.compile(checkpointer=saver)


class WorkflowRunner:
    def __init__(
        self,
        llm: LLMClient,
        *,
        checkpointer: Any | None = None,
        checkpoint_path: str | Path | None = None,
        vector_persist_directory: str | None = ".gtm-vector-store",
    ) -> None:
        if checkpointer is not None and checkpoint_path is not None:
            raise ValueError("provide either checkpointer or checkpoint_path, not both")
        self._checkpoint_connection: sqlite3.Connection | None = None
        if checkpoint_path is not None:
            checkpointer, self._checkpoint_connection = _sqlite_saver(checkpoint_path)
        self.graph = build_graph(
            llm,
            checkpointer=checkpointer,
            vector_persist_directory=vector_persist_directory,
        )

    def close(self) -> None:
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None

    def __enter__(self) -> WorkflowRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def start(
        self,
        brief: ProductBrief,
        campaign: CampaignRequest,
        *,
        documents: list[ProductBrief] | None = None,
        thread_id: str | None = None,
        max_revisions: int = 2,
        max_steps: int = 20,
    ) -> tuple[str, WorkflowState]:
        run_id = thread_id or str(uuid4())
        self.graph.invoke(
            {
                "brief": brief,
                "documents": documents or [brief],
                "campaign": campaign,
                "max_revisions": max_revisions,
                "max_steps": max_steps,
            },
            self.config(run_id),
        )
        return run_id, self.get_state(run_id)

    def resume(self, thread_id: str, response: HumanReviewResponse) -> WorkflowState:
        self.graph.invoke(Command(resume=response.model_dump(mode="json")), self.config(thread_id))
        return self.get_state(thread_id)

    def get_state(self, thread_id: str) -> WorkflowState:
        return self.graph.get_state(self.config(thread_id)).values
