from __future__ import annotations

from typing import Any, Protocol

from gtm_agent.models.schemas import (
    CampaignRequest,
    CampaignAnalysis,
    ContentSuite,
    EvidencePassage,
    ProductBrief,
    ReviewFinding,
    ReviewResult,
    SupervisorDecision,
)


class ProviderFailure(RuntimeError):
    """The upstream model provider could not complete a request."""


class MalformedModelOutput(RuntimeError):
    """A model response did not satisfy the required schema."""


class LLMClient(Protocol):
    def decide(self, state_summary: dict[str, Any]) -> SupervisorDecision: ...

    def analyze(
        self,
        brief: ProductBrief,
        campaign: CampaignRequest,
        evidence: list[EvidencePassage],
    ) -> CampaignAnalysis: ...

    def generate(
        self,
        brief: ProductBrief,
        campaign: CampaignRequest,
        evidence: list[EvidencePassage],
        revision_findings: list[ReviewFinding] | None = None,
        analysis: CampaignAnalysis | None = None,
    ) -> ContentSuite: ...

    def review(
        self,
        brief: ProductBrief,
        campaign: CampaignRequest,
        evidence: list[EvidencePassage],
        content: ContentSuite,
        deterministic_checks: dict[str, bool],
        analysis: CampaignAnalysis | None = None,
    ) -> ReviewResult: ...
