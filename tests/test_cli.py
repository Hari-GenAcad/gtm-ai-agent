import json
from pathlib import Path

from gtm_agent.cli import main
from gtm_agent.llm import FakeLLM, FakeScenario
from gtm_agent.models.schemas import ProductBrief
from gtm_agent.orchestration import WorkflowRunner


def write_brief(tmp_path: Path) -> Path:
    path = tmp_path / "brief.md"
    path.write_text(
        "LaunchPad helps revenue teams run campaigns.\nLaunch date: November 1\nPricing: INR 4,999",
        encoding="utf-8",
    )
    return path


def base_args(path: Path) -> list[str]:
    return [str(path), "--objective", "Launch to revenue teams", "--audience", "revenue teams"]


def test_cli_happy_path_exports_machine_readable_state(tmp_path, capsys) -> None:
    brief = write_brief(tmp_path)
    output = tmp_path / "result.json"
    assert main([*base_args(brief), "--output", str(output)]) == 0
    state = json.loads(output.read_text(encoding="utf-8"))
    assert state["status"] == "approved"
    assert len(state["content"]["ads"]) == 3
    assert "Exported:" in capsys.readouterr().out


def test_cli_provider_failure_returns_nonzero(tmp_path, capsys) -> None:
    brief = write_brief(tmp_path)
    assert main([*base_args(brief), "--scenario", "provider_failure"]) == 1
    state = json.loads(capsys.readouterr().out)
    assert state["status"] == "error"


def test_cli_can_complete_human_rejection_noninteractively(tmp_path, capsys) -> None:
    brief = write_brief(tmp_path)
    code = main(
        [
            *base_args(brief),
            "--scenario", "review_always_fails",
            "--max-revisions", "0",
            "--human-response", "reject",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert '"status": "unapproved"' in output
    assert '"stage": "failed"' in output


def test_cli_requires_provider_specific_keys(tmp_path, monkeypatch, capsys) -> None:
    brief = write_brief(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main([*base_args(brief), "--provider", "gemini"]) == 2
    assert "GEMINI_API_KEY is required" in capsys.readouterr().out
    assert main([*base_args(brief), "--provider", "openai"]) == 2
    assert "OPENAI_API_KEY is required" in capsys.readouterr().out


def test_cli_reports_invalid_input_without_traceback(tmp_path, capsys) -> None:
    path = tmp_path / "brief.docx"
    path.write_text("not supported", encoding="utf-8")
    assert main(base_args(path)) == 2
    assert "Unable to load brief: unsupported document type" in capsys.readouterr().out


def test_cli_accepts_public_google_sheet_as_only_source(monkeypatch, capsys) -> None:
    source_url = "https://docs.google.com/spreadsheets/d/public-sheet/edit"
    monkeypatch.setattr(
        "gtm_agent.cli.load_public_google_sheet",
        lambda url: [
            ProductBrief(
                title="Launch calendar",
                source_id=url,
                content="LaunchPad helps revenue teams run campaigns. Launch date: November 1.",
                media_type="spreadsheet",
            )
        ],
    )
    code = main(["--google-sheet", source_url, "--objective", "Launch LaunchPad"])
    assert code == 0
    assert '"status": "approved"' in capsys.readouterr().out


def test_cli_resume_requires_durable_checkpoint_arguments(capsys) -> None:
    assert main(["--resume-thread", "thread-1"]) == 2
    assert "requires --checkpoint-db and --human-response" in capsys.readouterr().out


def test_cli_resumes_checkpoint_created_by_previous_runner(tmp_path, campaign, capsys) -> None:
    database = tmp_path / "workflow.sqlite"
    brief = ProductBrief(
        title="LaunchPad",
        source_id="brief.md",
        content="LaunchPad helps revenue teams run campaigns. Launch date: November 1.",
    )
    first_runner = WorkflowRunner(
        FakeLLM(FakeScenario.REVIEW_ALWAYS_FAILS),
        checkpoint_path=database,
        vector_persist_directory=None,
    )
    thread_id, paused = first_runner.start(brief, campaign, max_revisions=0)
    first_runner.close()
    assert paused["status"] == "waiting_human"

    code = main(
        [
            "--resume-thread",
            thread_id,
            "--checkpoint-db",
            str(database),
            "--human-response",
            "approve",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert '"status": "unapproved"' in output
    assert f"Checkpoint thread: {thread_id}" in output
