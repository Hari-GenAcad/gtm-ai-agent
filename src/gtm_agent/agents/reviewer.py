from __future__ import annotations

from gtm_agent.llm.base import LLMClient
from gtm_agent.models.schemas import (
    CampaignRequest,
    CampaignAnalysis,
    ContentSuite,
    EvidencePassage,
    ProductBrief,
    ReviewFinding,
    ReviewResult,
)


def objective_checks(content: ContentSuite, evidence: list[EvidencePassage]) -> dict[str, bool]:
    source_ids = {item.source_id for item in evidence}
    assets = [content.linkedin, content.email, content.blog, *content.ads]
    evidence_text = " ".join(" ".join(item.text.casefold().split()) for item in evidence)
    return {
        "linkedin_present": bool(content.linkedin.body.strip()),
        "email_subject_and_body_present": bool(content.email.title.strip() and content.email.body.strip()),
        "blog_headline_and_body_present": bool(content.blog.title.strip() and content.blog.body.strip()),
        "three_distinct_ads_present": len(content.ads) >= 3
        and len({item.body.strip().casefold() for item in content.ads}) == len(content.ads),
        "all_assets_attributed": bool(source_ids)
        and all(bool(asset.source_ids) and set(asset.source_ids) <= source_ids for asset in assets),
        "verified_facts_grounded": bool(content.verified_facts)
        and all(" ".join(fact.casefold().split()) in evidence_text for fact in content.verified_facts),
    }


def review_content(
    llm: LLMClient,
    brief: ProductBrief,
    campaign: CampaignRequest,
    evidence: list[EvidencePassage],
    content: ContentSuite,
    analysis: CampaignAnalysis | None = None,
) -> ReviewResult:
    checks = objective_checks(content, evidence)
    result = llm.review(brief, campaign, evidence, content, checks, analysis)
    failed = [name for name, passed in checks.items() if not passed]
    if not failed:
        return result
    findings = list(result.findings)
    findings.extend(
        ReviewFinding(
            category="completeness" if "present" in name else "grounding",
            severity="error",
            message=f"Objective check failed: {name}.",
            action=f"Correct the {name.replace('_', ' ')} requirement.",
        )
        for name in failed
    )
    return ReviewResult(
        approved=False,
        decision="fail",
        summary=f"Objective validation failed: {', '.join(failed)}.",
        findings=findings,
        deterministic_checks=checks,
    )
