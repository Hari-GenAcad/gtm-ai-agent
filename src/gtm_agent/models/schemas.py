from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WorkflowStage(StrEnum):
    INITIALIZE = "initialize"
    SUPERVISE = "supervise"
    RETRIEVE = "retrieve"
    ANALYZE = "analyze"
    GENERATE = "generate"
    REVIEW = "review"
    REVISE = "revise"
    HUMAN_REVIEW = "human_review"
    COMPLETE = "complete"
    FAILED = "failed"


class WorkflowStatus(StrEnum):
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    APPROVED = "approved"
    UNAPPROVED = "unapproved"
    ERROR = "error"


class SupervisorActionName(StrEnum):
    RETRIEVE = "retrieve"
    ANALYZE = "analyze"
    GENERATE = "generate"
    REVIEW = "review"
    REVISE = "revise"
    REQUEST_HUMAN = "request_human"
    COMPLETE = "complete"
    FAIL = "fail"


class ErrorKind(StrEnum):
    PROVIDER = "provider_failure"
    MALFORMED_OUTPUT = "malformed_output"
    RETRIEVAL = "insufficient_evidence"
    INVALID_ACTION = "invalid_supervisor_action"
    VALIDATION = "validation_failure"
    LIMIT = "limit_reached"
    INTERNAL = "internal_error"


class ProductBrief(StrictModel):
    title: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    media_type: Literal["text", "markdown", "pdf", "spreadsheet", "notion_export"] = "text"


class CampaignRequest(StrictModel):
    objective: str = Field(min_length=1)
    audience: str = ""
    tone: str = ""


class CampaignAnalysis(StrictModel):
    target_audience: list[str] = Field(min_length=1)
    tone: str = Field(min_length=1)
    value_proposition: str = Field(min_length=1)
    key_messages: list[str] = Field(min_length=1)
    missing_facts: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class EvidencePassage(StrictModel):
    passage_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float = Field(ge=0)
    chunk_index: int = Field(ge=0)


class ContentAsset(StrictModel):
    kind: Literal["linkedin", "email", "blog", "ad"]
    title: str = ""
    body: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class ContentSuite(StrictModel):
    linkedin: ContentAsset
    email: ContentAsset
    blog: ContentAsset
    ads: list[ContentAsset] = Field(min_length=3)
    verified_facts: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_asset_kinds(self) -> ContentSuite:
        expected = {
            "linkedin": self.linkedin.kind,
            "email": self.email.kind,
            "blog": self.blog.kind,
        }
        wrong = [name for name, kind in expected.items() if name != kind]
        if wrong or any(ad.kind != "ad" for ad in self.ads):
            raise ValueError("content assets must use their matching kind")
        return self


class ReviewFinding(StrictModel):
    category: Literal[
        "grounding", "consistency", "tone", "audience", "completeness", "missing_fact"
    ]
    severity: Literal["info", "warning", "error"]
    message: str = Field(min_length=1)
    action: str = Field(min_length=1)


class ReviewResult(StrictModel):
    approved: bool
    decision: Literal["pass", "fail"]
    summary: str = Field(min_length=1)
    findings: list[ReviewFinding] = Field(default_factory=list)
    deterministic_checks: dict[str, bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def approval_matches_decision(self) -> ReviewResult:
        if self.approved != (self.decision == "pass"):
            raise ValueError("approved and decision must agree")
        return self


class SupervisorDecision(StrictModel):
    action: SupervisorActionName
    reason: str = Field(min_length=1)


class HumanReviewRequest(StrictModel):
    reason: str = Field(min_length=1)
    questions: list[str] = Field(min_length=1)
    unresolved_findings: list[str] = Field(default_factory=list)


class HumanReviewResponse(StrictModel):
    decision: Literal["approve", "reject", "clarify"]
    comment: str = ""
    factual_updates: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def response_has_context(self) -> HumanReviewResponse:
        if self.decision != "approve" and not (self.comment or self.factual_updates):
            raise ValueError("reject and clarify responses require context")
        return self


class WorkflowError(StrictModel):
    kind: ErrorKind
    message: str = Field(min_length=1)
    recoverable: bool = False
    attempt: int = Field(default=1, ge=1)


class ActionRecord(StrictModel):
    step: int = Field(ge=0)
    action: str = Field(min_length=1)
    outcome: Literal["selected", "success", "retry", "paused", "rejected", "error"]
    detail: str = Field(min_length=1)


class WorkflowState(TypedDict, total=False):
    brief: ProductBrief
    documents: list[ProductBrief]
    campaign: CampaignRequest
    analysis: CampaignAnalysis | None
    stage: WorkflowStage
    status: WorkflowStatus
    evidence: list[EvidencePassage]
    content: ContentSuite | None
    review: ReviewResult | None
    review_history: list[ReviewResult]
    revision_count: int
    max_revisions: int
    step_count: int
    max_steps: int
    provider_attempts: int
    max_provider_attempts: int
    decision: SupervisorDecision | None
    history: list[ActionRecord]
    errors: list[WorkflowError]
    human_request: HumanReviewRequest | None
    human_response: HumanReviewResponse | None
    metadata: dict[str, Any]
