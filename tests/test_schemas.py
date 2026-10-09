import pytest
from pydantic import ValidationError

from gtm_agent.models.schemas import (
    ContentAsset,
    ContentSuite,
    HumanReviewResponse,
    ReviewResult,
)


def asset(kind: str, body: str = "copy") -> ContentAsset:
    return ContentAsset(kind=kind, title="title", body=body, source_ids=["brief.md"])


def test_content_suite_requires_three_ads() -> None:
    with pytest.raises(ValidationError):
        ContentSuite(
            linkedin=asset("linkedin"),
            email=asset("email"),
            blog=asset("blog"),
            ads=[asset("ad"), asset("ad")],
        )


def test_content_suite_validates_asset_kinds() -> None:
    with pytest.raises(ValidationError, match="matching kind"):
        ContentSuite(
            linkedin=asset("email"),
            email=asset("email"),
            blog=asset("blog"),
            ads=[asset("ad", str(index)) for index in range(3)],
        )


def test_review_decision_must_match_approval() -> None:
    with pytest.raises(ValidationError, match="must agree"):
        ReviewResult(approved=True, decision="fail", summary="inconsistent")


def test_reject_and_clarify_require_context() -> None:
    with pytest.raises(ValidationError, match="require context"):
        HumanReviewResponse(decision="reject")
    assert HumanReviewResponse(decision="approve").decision == "approve"

