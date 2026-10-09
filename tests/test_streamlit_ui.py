from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_upload_runs_complete_fake_workflow() -> None:
    app_path = Path("src/gtm_agent/ui/streamlit_app.py").resolve()
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert app.button[0].disabled
    app.get("file_uploader")[0].upload(
        "brief.md",
        b"LaunchPad helps revenue teams run campaigns. Launch date: November 1. Pricing: INR 4,999.",
        "text/markdown",
    ).run(timeout=30)
    assert not app.button[0].disabled

    app.button[0].click().run(timeout=60)

    assert not app.exception
    assert any(item.label == "Workflow status" and item.value == "Approved" for item in app.metric)
    assert any(item.label == "Retrieval" and item.value == "Vector" for item in app.metric)
    assert any(tab.label == "LinkedIn" for tab in app.tabs)
    assert any(tab.label == "Ads" for tab in app.tabs)
    assert "workflow_state" in app.session_state
    assert len(app.session_state["workflow_state"]["content"].ads) == 3
    assert app.session_state["checkpoint_thread"]
    app.session_state["runner"].close()
