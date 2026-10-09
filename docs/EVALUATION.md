# Evaluation Report

## Final automated result

- Date: 2026-10-09
- Environment: isolated `D:\gtm-ai-agent\.venv-clean` (no system site-packages)
- Dependency health: `pip check` -> `No broken requirements found.`
- Test command: `.\.venv-clean\Scripts\python.exe -m pytest --cov=gtm_agent --cov-report=term-missing`
- Acceptance result: **69 passed, 0 failed, 0 skipped**
- Instrumented core coverage: **94%**
- Orchestration coverage: **96%**
- Streamlit is run through `AppTest` in a separate script runner; that passing end-to-end flow is not attributed by the parent coverage process.

## Verified input and retrieval behavior

- UTF-8 text/Markdown normalization and real PDF extraction.
- CSV/XLSX spreadsheet rows and sheet names.
- Public Google Sheets URL validation, CSV export transformation, and URL attribution.
- Rejection of HTTP, non-Google, non-spreadsheet, and invalid-`gid` URLs.
- Notion ZIP expansion into separately attributed Markdown/CSV documents.
- Multiple-source merge, persistent Chroma reuse, local MiniLM embeddings, semantic paraphrase retrieval, and explicit BM25 fallback metadata.

## Verified workflow behavior

- Audience/tone inference and preservation of user overrides.
- Value proposition, key messages, rationale, and missing facts.
- LinkedIn, email, blog, and three distinct ads with attributed evidence.
- Happy completion and failed-review -> revision -> approval.
- Revision exhaustion -> human interrupt.
- Human approval, rejection, clarification, factual-update retention, and resume.
- SQLite checkpoint close, new runner/process boundary, state reload, and resume.
- CLI resume from a checkpoint created by a previous runner.
- Missing, contradictory, insufficient, and risky-evidence gates.
- Provider failures across supervisor, analysis, generation, review, and revision.
- Malformed output, illegal supervisor action, retry exhaustion, and step-limit termination.
- Complete Streamlit upload-to-approved fake-provider flow.

## Evidence-safety cases

| Case | Observed result |
|---|---|
| Complete brief | Four-format suite approved with citations |
| Missing launch date | Timing not invented; missing fact flagged |
| Contradictory dates | Paused before generation |
| Unsupported superlative/guarantee | Paused before repeating claim |
| Insufficient evidence | Human clarification requested |

## Saved end-to-end sample

Three local sources were indexed with MiniLM/Chroma: `product_brief.md`, `launch_calendar.csv`, and `past_campaign.md`. With audience and tone omitted, the agent selected product marketing managers/revenue leaders and a clear, confident, practical tone. The deterministic review failed once, triggered one revision, then passed. Exact output is in `examples/sample_content_suite.json`.

## Live-provider status

Gemini authentication, structured generation, and review were verified with the sample LaunchPilot AI PDF. The final live run completed with vector retrieval, two evidence passages, all four required formats, three ads, one passing review, zero revisions, and zero errors. Temporary free-tier capacity failures remain possible and are handled with bounded retries and an explicit timeout.

## Remaining limitations

- Private/authenticated Google Sheets and direct Notion APIs need user OAuth/service credentials; public Sheets and exports are supported.
- Research uses approved sources instead of unrestricted web search.
- Evidence conflict detection is intentionally narrow.
- UI restart recovery uses the displayed thread ID with the CLI resume interface.
- The regression set does not establish production-level copy quality.
