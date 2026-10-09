import os

import pytest

from gtm_agent.models.schemas import CampaignRequest, ProductBrief


os.environ.setdefault("GTM_EMBEDDING_BACKEND", "hash")
os.environ.setdefault("GTM_EMBEDDINGS_LOCAL_ONLY", "true")


@pytest.fixture
def brief() -> ProductBrief:
    return ProductBrief(
        title="LaunchPad",
        source_id="brief.md",
        content=(
            "LaunchPad helps revenue teams turn approved product facts into coordinated campaign drafts.\n\n"
            "Launch date: November 1, 2026\nPricing: INR 4,999 per workspace per month.\n\n"
            "A human editor must approve every draft before publication."
        ),
        media_type="markdown",
    )


@pytest.fixture
def campaign() -> CampaignRequest:
    return CampaignRequest(
        objective="Launch an evidence-grounded campaign for revenue teams",
        audience="revenue teams",
        tone="clear and confident",
    )
