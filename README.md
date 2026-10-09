# GTM Content Agent

An evidence-grounded LangGraph agent that ingests product, launch-calendar, event, and past-campaign documents; indexes them in a persistent local vector store; derives campaign positioning; generates a four-format content suite; reviews it; and conditionally revises or pauses for a human.

## Implemented capabilities

- PDF, text/Markdown, CSV/XLSX, public Google Sheets URL, and Notion Markdown/CSV ZIP ingestion.
- Local `all-MiniLM-L6-v2` embeddings and persistent Chroma vector retrieval.
- Filename, URL, passage, chunk, and similarity-score attribution.
- Explicit BM25 fallback if embeddings or Chroma are unavailable.
- Audience, tone, value proposition, key-message, and missing-fact analysis.
- Optional audience and tone overrides.
- LinkedIn post, promotional email, short blog, and at least three ad variations.
- Factual, cross-format consistency, tone, audience, and completeness review.
- LangGraph conditional routing with bounded revision, provider retry, step limits, and human interrupt/resume.
- Optional SQLite checkpoints that survive process restarts.
- Deterministic offline scenarios plus Gemini and OpenAI structured-output adapters.
- CLI, Streamlit UI, JSON export, and a saved three-source sample suite.

## Setup

Python 3.11-3.13 is recommended. From `D:\gtm-ai-agent`:

```powershell
python -m venv .venv-clean
$env:PIP_CACHE_DIR = "D:\gtm-ai-agent\.pip-cache"
.\.venv-clean\Scripts\python.exe -m pip install -e ".[dev,ui]"
.\.venv-clean\Scripts\python.exe -m pip check
```

The clean environment is isolated from global packages and keeps its environment and pip cache on D:. On a new machine, allow the initial embedding-model download and set its cache to D::

```powershell
$env:HF_HOME = "D:\gtm-ai-agent\.models"
$env:GTM_EMBEDDINGS_LOCAL_ONLY = "false"
```

After the first download, restore `GTM_EMBEDDINGS_LOCAL_ONLY=true` for offline operation.

## Run the complete offline sample

Audience and tone are intentionally omitted so the agent must infer them:

```powershell
.\.venv-clean\Scripts\gtm-agent.exe `
  examples\product_brief.md `
  examples\launch_calendar.csv `
  examples\past_campaign.md `
  --objective "Create a coordinated launch campaign for LaunchPad" `
  --scenario review_fail_then_pass `
  --output examples\sample_content_suite.json
```

Verified result: vector retrieval across three sources, inferred positioning, all four formats with three ads, one bounded revision, and final `approved` status. See [sample_content_suite.json](examples/sample_content_suite.json) and [SAMPLE_OUTPUT.md](examples/SAMPLE_OUTPUT.md).

## Public Google Sheets

The sheet must be shared for public read access. Only HTTPS `docs.google.com/spreadsheets` URLs are accepted; arbitrary URLs are rejected.

```powershell
.\.venv-clean\Scripts\gtm-agent.exe `
  --google-sheet "https://docs.google.com/spreadsheets/d/SHEET_ID/edit#gid=0" `
  --objective "Create the launch campaign"
```

Private Google Sheets and direct Notion workspaces require user OAuth/service credentials. Those authenticated connectors are not claimed; CSV/XLSX and Notion ZIP exports remain fully supported.

## Restart-safe human review

Start a durable run with a stable ID:

```powershell
.\.venv-clean\Scripts\gtm-agent.exe examples\product_brief.md `
  --objective "Launch LaunchPad" `
  --scenario review_always_fails `
  --checkpoint-db .gtm-checkpoints.sqlite `
  --thread-id demo-1
```

If the process stops at human review, another process can resume the saved state:

```powershell
.\.venv-clean\Scripts\gtm-agent.exe `
  --checkpoint-db .gtm-checkpoints.sqlite `
  --resume-thread demo-1 `
  --human-response approve
```

Fake scenarios are `happy`, `review_fail_then_pass`, `review_always_fails`, `missing_evidence`, `malformed_output`, `provider_failure`, and `invalid_supervisor_action`.

## Streamlit UI

```powershell
.\.venv-clean\Scripts\python.exe -m streamlit run src\gtm_agent\ui\streamlit_app.py
```

Upload source files or provide a public Google Sheets URL. The tabs expose evidence, campaign analysis, content, reviews, revision history, and agent actions. UI runs use `.gtm-checkpoints.sqlite` and display their checkpoint thread ID.

## Real providers

The ignored local `.env` supports `GEMINI_API_KEY`, optional `GEMINI_MODEL`, `OPENAI_API_KEY`, and `GTM_EMBEDDINGS_LOCAL_ONLY=true`. Run with `--provider gemini` or `--provider openai`.

Gemini authentication previously reached the service, but attempted free-tier models returned HTTP 503 high-demand responses. Live success is not claimed; provider contracts and bounded failure recovery are tested offline.

## Tests

```powershell
.\.venv-clean\Scripts\python.exe -m pytest
.\.venv-clean\Scripts\python.exe -m pytest --cov=gtm_agent --cov-report=term-missing
```

Final acceptance result on 2026-10-09:

- 68 passed, 0 failed, 0 skipped.
- 94% instrumented core coverage; the Streamlit flow is exercised with `AppTest` in a separate script runner.
- 96% orchestration coverage.
- Clean `pip check`: no broken requirements.
- Semantic MiniLM retrieval, SQLite close/reopen resume, CLI restart/resume, and Streamlit upload-to-approved behavior are verified.

See [EVALUATION.md](docs/EVALUATION.md).

## Workflow

```text
START -> initialize -> supervisor
                         |-> retrieve -> supervisor
                         |-> analyze  -> supervisor
                         |-> generate -> supervisor
                         |-> review   -> supervisor
                         |-> revise   -> supervisor
                         |-> human interrupt -> supervisor
                         |-> complete -> END
                         `-> fail -> END
```

Every model-selected action is schema-constrained and checked against actions legal for the current state.

## Known limitations

- Product research is confined to approved inputs; unrestricted web research is excluded to preserve grounding.
- Private Sheets/direct Notion APIs need credentials and consent not supplied to this project.
- The evidence-risk detector catches selected labelled contradictions and risky guarantees, not every possible factual conflict.
- The UI displays a durable thread ID, but after a full UI server restart recovery currently uses the CLI resume command.
- The deterministic sample proves the application flow, not live-model writing quality.
- External publishing and production deployment remain out of scope.
