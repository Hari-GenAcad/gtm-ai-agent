import json
from pathlib import Path

from gtm_agent.llm.fake import FakeLLM
from gtm_agent.models.schemas import CampaignRequest, ProductBrief, WorkflowStatus
from gtm_agent.orchestration import WorkflowRunner


CASES = json.loads((Path(__file__).parent / "fixtures" / "evaluation_cases.json").read_text())


def test_five_explicit_evaluation_cases() -> None:
    observed = {}
    for case in CASES:
        runner = WorkflowRunner(FakeLLM())
        _, state = runner.start(
            ProductBrief(title="LaunchPad", source_id=f"{case['id']}.md", content=case["brief"]),
            CampaignRequest(
                objective=case["objective"],
                audience="revenue teams",
                tone="clear and confident",
            ),
        )
        observed[case["id"]] = state["status"]

    assert len(CASES) == 5
    assert observed["complete-brief"] == WorkflowStatus.APPROVED
    assert observed["missing-launch-date"] == WorkflowStatus.APPROVED
    assert observed["contradictory-facts"] == WorkflowStatus.WAITING_HUMAN
    assert observed["unsupported-claim"] == WorkflowStatus.WAITING_HUMAN
    assert observed["insufficient-evidence"] == WorkflowStatus.WAITING_HUMAN

