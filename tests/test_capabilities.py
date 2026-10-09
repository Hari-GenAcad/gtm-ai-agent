import pytest

from gtm_agent.agents.reviewer import objective_checks, review_content
from gtm_agent.agents.supervisor import validate_decision
from gtm_agent.llm.base import MalformedModelOutput
from gtm_agent.llm.fake import FakeLLM
from gtm_agent.llm.openai_client import OpenAILLM
from gtm_agent.models.schemas import (
    ContentAsset,
    EvidencePassage,
    SupervisorActionName,
    SupervisorDecision,
)


def evidence() -> list[EvidencePassage]:
    return [
        EvidencePassage(
            passage_id="brief.md#chunk-1",
            source_id="brief.md",
            text="LaunchPad helps revenue teams launch campaigns.",
            score=1,
            chunk_index=0,
        )
    ]


def test_fake_generation_and_objective_review(brief, campaign) -> None:
    client = FakeLLM()
    suite = client.generate(brief, campaign, evidence())
    checks = objective_checks(suite, evidence())
    assert all(checks.values())
    result = review_content(client, brief, campaign, evidence(), suite)
    assert result.approved


def test_objective_review_rejects_bad_attribution(brief, campaign) -> None:
    client = FakeLLM()
    suite = client.generate(brief, campaign, evidence())
    suite.linkedin = ContentAsset(kind="linkedin", title="Post", body="copy", source_ids=["other.md"])
    result = review_content(client, brief, campaign, evidence(), suite)
    assert not result.approved
    assert result.deterministic_checks["all_assets_attributed"] is False


def test_supervisor_allowlist_rejects_wrong_state_action(brief, campaign) -> None:
    state = {
        "brief": brief,
        "campaign": campaign,
        "metadata": {},
        "step_count": 0,
        "max_steps": 20,
    }
    with pytest.raises(MalformedModelOutput, match="invalid for current state"):
        validate_decision(
            SupervisorDecision(action=SupervisorActionName.COMPLETE, reason="too early"), state
        )


def test_supervisor_accepts_retrieve_as_first_action(brief, campaign) -> None:
    state = {"brief": brief, "campaign": campaign, "metadata": {}}
    decision = SupervisorDecision(action=SupervisorActionName.RETRIEVE, reason="need evidence")
    assert validate_decision(decision, state) == decision


def test_openai_adapter_requests_structured_output_without_network(brief, campaign) -> None:
    expected = FakeLLM().generate(brief, campaign, evidence())

    class Responses:
        def parse(self, **kwargs):
            assert kwargs["model"] == "test-model"
            assert kwargs["text_format"].__name__ == "ContentSuite"
            return type("Response", (), {"output_parsed": expected})()

    client = type("Client", (), {"responses": Responses()})()
    adapter = OpenAILLM(model="test-model", client=client)
    assert adapter.generate(brief, campaign, evidence()) == expected
