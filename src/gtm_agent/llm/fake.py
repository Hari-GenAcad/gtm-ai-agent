from __future__ import annotations

from enum import StrEnum
from typing import Any

from gtm_agent.llm.base import MalformedModelOutput, ProviderFailure
from gtm_agent.models.schemas import (
    CampaignRequest,
    CampaignAnalysis,
    ContentAsset,
    ContentSuite,
    EvidencePassage,
    ProductBrief,
    ReviewFinding,
    ReviewResult,
    SupervisorActionName,
    SupervisorDecision,
)


class FakeScenario(StrEnum):
    HAPPY = "happy"
    REVIEW_FAIL_THEN_PASS = "review_fail_then_pass"
    REVIEW_ALWAYS_FAILS = "review_always_fails"
    MISSING_EVIDENCE = "missing_evidence"
    MALFORMED_OUTPUT = "malformed_output"
    PROVIDER_FAILURE = "provider_failure"
    INVALID_SUPERVISOR_ACTION = "invalid_supervisor_action"


class FakeLLM:
    """Scenario-driven offline client with explicit, deterministic behavior."""

    def __init__(self, scenario: FakeScenario | str = FakeScenario.HAPPY) -> None:
        self.scenario = FakeScenario(scenario)
        self.review_calls = 0
        self.generate_calls = 0
        self.decide_calls = 0

    def decide(self, state_summary: dict[str, Any]) -> SupervisorDecision:
        self.decide_calls += 1
        if self.scenario == FakeScenario.INVALID_SUPERVISOR_ACTION and self.decide_calls == 1:
            return {"action": "publish_everywhere", "reason": "invalid fixture"}  # type: ignore[return-value]

        if state_summary["fatal_error"] or state_summary["human_rejected"]:
            action, reason = SupervisorActionName.FAIL, "A terminal error or explicit rejection stops the run."
        elif state_summary["human_override"]:
            action, reason = SupervisorActionName.COMPLETE, "A human explicitly accepted the flagged result."
        elif not state_summary["retrieval_attempted"]:
            action, reason = SupervisorActionName.RETRIEVE, "Evidence has not been retrieved."
        elif not state_summary["has_evidence"]:
            action, reason = SupervisorActionName.REQUEST_HUMAN, "No relevant source evidence was found."
        elif state_summary["evidence_issues"]:
            action, reason = SupervisorActionName.REQUEST_HUMAN, "Conflicting or risky source claims need review."
        elif not state_summary["has_analysis"]:
            action, reason = SupervisorActionName.ANALYZE, "Evidence is ready for campaign analysis."
        elif not state_summary["has_content"]:
            action, reason = SupervisorActionName.GENERATE, "Grounded evidence is ready for generation."
        elif not state_summary["has_review"]:
            action, reason = SupervisorActionName.REVIEW, "Generated assets need independent review."
        elif state_summary["review_approved"]:
            action, reason = SupervisorActionName.COMPLETE, "The latest review approved the suite."
        elif state_summary["revision_count"] < state_summary["max_revisions"]:
            action, reason = SupervisorActionName.REVISE, "Review findings require a bounded revision."
        else:
            action, reason = SupervisorActionName.REQUEST_HUMAN, "Automated revisions are exhausted."
        return SupervisorDecision(action=action, reason=reason)

    def analyze(
        self,
        brief: ProductBrief,
        campaign: CampaignRequest,
        evidence: list[EvidencePassage],
    ) -> CampaignAnalysis:
        text = " ".join(item.text for item in evidence) or brief.content
        lower = text.casefold()
        known_audiences = [
            label
            for needle, label in [
                ("product marketing", "product marketing managers"),
                ("revenue", "revenue leaders"),
                ("founder", "founders"),
                ("consultant", "consultants"),
                ("executive", "executives"),
                ("student", "students"),
            ]
            if needle in lower
        ]
        audience = [campaign.audience] if campaign.audience else known_audiences or ["prospective users"]
        first_fact = " ".join(text.replace("#", "").split())[:300]
        missing: list[str] = []
        if "launch" not in lower and "available" not in lower:
            missing.append("Launch timing is not provided.")
        if "price" not in lower and "pricing" not in lower:
            missing.append("Pricing is not provided.")
        return CampaignAnalysis(
            target_audience=audience,
            tone=campaign.tone or "clear, confident, and practical",
            value_proposition=first_fact,
            key_messages=[" ".join(item.text.split())[:220] for item in evidence[:3]] or [first_fact],
            missing_facts=missing,
            rationale="Audience, tone, and messages were derived from retrieved approved evidence.",
        )

    def generate(
        self,
        brief: ProductBrief,
        campaign: CampaignRequest,
        evidence: list[EvidencePassage],
        revision_findings: list[ReviewFinding] | None = None,
        analysis: CampaignAnalysis | None = None,
    ) -> ContentSuite:
        self.generate_calls += 1
        if self.scenario == FakeScenario.PROVIDER_FAILURE:
            raise ProviderFailure("deterministic provider failure")
        if self.scenario == FakeScenario.MALFORMED_OUTPUT:
            raise MalformedModelOutput("deterministic malformed structured output")
        if not evidence:
            raise MalformedModelOutput("generation requires at least one evidence passage")

        sources = sorted({item.source_id for item in evidence})
        source_label = ", ".join(sources)
        fact = " ".join(evidence[0].text.split())[:280]
        revision_note = " Revised to address review feedback." if revision_findings else ""
        product_name = brief.title.removesuffix(" product brief").removesuffix(" Product Brief")
        positioning = analysis.value_proposition[:220] if analysis else fact
        citation = f"[Sources: {source_label}]"
        shared = f"{positioning} {citation}"
        missing: list[str] = []
        lower = brief.content.lower()
        if "launch" not in lower and "available" not in lower:
            missing.append("Launch timing is not provided in the approved source.")
        if "price" not in lower and "pricing" not in lower:
            missing.append("Pricing is not provided in the approved source.")

        return ContentSuite(
            linkedin=ContentAsset(
                kind="linkedin",
                title="LinkedIn post",
                body=(
                    f"Campaign planning should start with approved facts—not a blank page.\n\n"
                    f"Meet {product_name}. {shared}\n\n"
                    f"Built for {campaign.audience}, it provides a consistent, ready-to-edit campaign starting point. "
                    f"{campaign.objective}.{revision_note}\n\nReview the evidence, refine the voice, and launch with confidence."
                ),
                source_ids=sources,
            ),
            email=ContentAsset(
                kind="email",
                title=f"Introducing {product_name}: grounded campaign drafts from approved facts",
                body=(
                    f"Hello,\n\nCreating a coordinated launch campaign should not mean rewriting the same facts "
                    f"for every channel.\n\n{product_name} helps {campaign.audience} start from one grounded campaign story. "
                    f"{shared}\n\n{campaign.objective}.{revision_note}\n\nReview the draft suite and shape it for your launch."
                ),
                source_ids=sources,
            ),
            blog=ContentAsset(
                kind="blog",
                title=f"How {product_name} creates a grounded campaign starting point",
                body=(
                    f"## The challenge\n\nProduct launches need consistent facts across social, email, blog, and ads. "
                    f"Disconnected drafting makes that consistency difficult.\n\n"
                    f"## A grounded starting point\n\n{shared}\n\n"
                    f"## Why it matters\n\nFor {campaign.audience}, the result is a coordinated first draft that remains "
                    f"ready for human editing and approval. {campaign.objective}.{revision_note}"
                ),
                source_ids=sources,
            ),
            ads=[
                ContentAsset(kind="ad", title=f"Ad {index}", body=body, source_ids=sources)
                for index, body in enumerate(
                    [
                        f"Turn approved product facts into a coordinated campaign draft with {product_name}.",
                        f"One grounded story. Four campaign formats. Meet {product_name}.",
                        f"Create a consistent launch starting point for {campaign.audience} with {product_name}.",
                    ],
                    start=1,
                )
            ],
            verified_facts=[fact],
            missing_facts=missing,
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
        self.review_calls += 1
        objective_pass = all(deterministic_checks.values())
        forced_failure = self.scenario == FakeScenario.REVIEW_ALWAYS_FAILS or (
            self.scenario == FakeScenario.REVIEW_FAIL_THEN_PASS and self.review_calls == 1
        )
        approved = objective_pass and not forced_failure
        findings: list[ReviewFinding] = []
        if forced_failure:
            findings.append(
                ReviewFinding(
                    category="tone",
                    severity="error",
                    message="The deterministic scenario requests a tone revision.",
                    action=f"Revise every asset to make the {campaign.tone} tone more explicit.",
                )
            )
        return ReviewResult(
            approved=approved,
            decision="pass" if approved else "fail",
            summary="All grounding and consistency criteria passed." if approved else "Revision is required.",
            findings=findings,
            deterministic_checks=deterministic_checks,
        )
