from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from gtm_agent.llm.base import MalformedModelOutput, ProviderFailure
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


T = TypeVar("T", bound=BaseModel)


class OpenAILLM:
    """Structured-output OpenAI Responses adapter with no filesystem or tool access."""

    def __init__(self, *, model: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-6-astra")
        self.client = client or OpenAI()

    def _parse(self, *, system: str, payload: dict[str, Any], schema: type[T]) -> T:
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text_format=schema,
            )
        except OpenAIError as exc:
            raise ProviderFailure(str(exc)) from exc
        except (TypeError, ValueError, ValidationError, AttributeError, IndexError) as exc:
            raise MalformedModelOutput(str(exc)) from exc
        parsed = response.output_parsed
        if not isinstance(parsed, schema):
            raise MalformedModelOutput(f"provider did not return {schema.__name__}")
        return parsed

    def decide(self, state_summary: dict[str, Any]) -> SupervisorDecision:
        return self._parse(
            system=(
                "Choose exactly one allowed next workflow action from the SupervisorDecision schema. "
                "Never skip evidence retrieval or review, never approve a failed review, and request human "
                "input when evidence is absent or revision limits are exhausted."
            ),
            payload=state_summary,
            schema=SupervisorDecision,
        )

    def analyze(
        self,
        brief: ProductBrief,
        campaign: CampaignRequest,
        evidence: list[EvidencePassage],
    ) -> CampaignAnalysis:
        return self._parse(
            system=(
                "Analyze the approved product evidence. Identify a specific target audience, select a suitable "
                "tone, summarize the value proposition and key messages, and flag missing facts. Preserve any "
                "user-supplied audience or tone as constraints. Do not invent facts."
            ),
            payload={
                "brief": brief.model_dump(mode="json"),
                "campaign": campaign.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            schema=CampaignAnalysis,
        )

    def generate(
        self,
        brief: ProductBrief,
        campaign: CampaignRequest,
        evidence: list[EvidencePassage],
        revision_findings: list[ReviewFinding] | None = None,
        analysis: CampaignAnalysis | None = None,
    ) -> ContentSuite:
        return self._parse(
            system=(
                "Create all four GTM formats. Use only supplied evidence for factual claims, include source_ids "
                "on every asset, keep verified facts separate, and flag missing facts instead of inventing them. "
                "Every verified_facts entry must be a short, verbatim, contiguous quote from the supplied evidence; "
                "do not paraphrase verified_facts. Write polished channel-specific copy rather than copying raw passages."
            ),
            payload={
                "brief": brief.model_dump(mode="json"),
                "campaign": campaign.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "revision_findings": [item.model_dump(mode="json") for item in revision_findings or []],
                "campaign_analysis": analysis.model_dump(mode="json") if analysis else None,
            },
            schema=ContentSuite,
        )

    def review(
        self,
        brief: ProductBrief,
        campaign: CampaignRequest,
        evidence: list[EvidencePassage],
        content: ContentSuite,
        deterministic_checks: dict[str, bool],
        analysis: CampaignAnalysis | None = None,
    ) -> ReviewResult:
        return self._parse(
            system=(
                "Independently review grounding, cross-format consistency, tone, audience relevance, completeness, "
                "unsupported claims, and missing critical information. Any failed deterministic check requires fail."
            ),
            payload={
                "brief": brief.model_dump(mode="json"),
                "campaign": campaign.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "content": content.model_dump(mode="json"),
                "deterministic_checks": deterministic_checks,
                "campaign_analysis": analysis.model_dump(mode="json") if analysis else None,
            },
            schema=ReviewResult,
        )
