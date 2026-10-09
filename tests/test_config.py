import os

from gtm_agent.config import load_local_env


def test_load_local_env_loads_values_without_overriding_existing(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text("NEW_VALUE=loaded\nEXISTING=from-file\n# ignored\nINVALID\n", encoding="utf-8")
    monkeypatch.delenv("NEW_VALUE", raising=False)
    monkeypatch.setenv("EXISTING", "from-process")
    load_local_env(path)
    assert os.environ["NEW_VALUE"] == "loaded"
    assert os.environ["EXISTING"] == "from-process"


def test_load_local_env_missing_file_is_noop(tmp_path) -> None:
    load_local_env(tmp_path / "missing.env")
