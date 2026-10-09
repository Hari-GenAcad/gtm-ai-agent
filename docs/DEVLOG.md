# Development Log

## 2026-10-09 — M0 baseline and architecture

Objective: inspect the workspace, confirm the stack, and freeze a small architecture.

- Initial workspace contained only `.vscode/extensions.json`; it was not a Git repository.
- Runtime: Python 3.11.9 (`python`), with another Python 3.14 launcher entry; Node 22.20 and Git 2.54 are available.
- Installed libraries detected: Pydantic 2.13.4, LangGraph 1.0.10, LangChain Core 1.2.17, OpenAI 2.26.0, Streamlit 1.56.0, and rank-bm25 0.2.2. Pytest and pypdf were not initially installed.
- `OPENAI_API_KEY` was not configured; live inference is therefore unverified. Offline development will use explicit fake scenarios.
- The official Codex manual was fetched only to verify session configuration guidance. Project files remain restricted to the workspace.
- Architecture frozen as documented in `IMPLEMENTATION_PLAN.md`. Lexical BM25 was selected as the reliable local retrieval fallback and will not be described as vector search.
- Baseline tests: none existed.

Remaining after M0: implement M1–M5, install the project/dev dependencies into a D:-hosted virtual environment, run tests/evaluations, and record observed results.

## 2026-10-09 — M1 schemas and state

Objective: implement only state and validation contracts.

- Added strict Pydantic models for briefs, campaign context, evidence, assets, content suite, review, supervisor decisions, errors, action history, and human responses.
- Added typed LangGraph workflow state with concrete fields used by routing and interfaces.
- Focused schema tests now reject missing ads, mismatched asset kinds, inconsistent review decisions, and context-free rejection/clarification.

## 2026-10-09 — M2 capabilities

Objective: implement provider-independent capabilities and deterministic fixtures.

- Added text/Markdown normalization and optional PDF parsing.
- Added attributed local BM25 retrieval and narrow checks for labelled contradictions and risky guarantees/superlatives.
- Added a provider protocol, seven fake scenarios, and an OpenAI Responses structured-output adapter.
- Added deterministic content completeness, uniqueness, source-attribution, and verified-fact grounding checks.
- A first test run exposed a whitespace mismatch between generated verified facts and evidence. Normalizing whitespace in both operands fixed the real grounding defect; the check was not disabled.

## 2026-10-09 — M3 graph orchestration

Objective: implement state-dependent routing and bounded failure recovery.

- Built the LangGraph topology with conditional supervisor edges for all allowed actions.
- Legal actions depend on current evidence/content/review/limit/error/human state.
- Added two-attempt provider/structured-output recovery, two-revision default, and 20-step default.
- Verified review failure routes through revision and returns for another review.

## 2026-10-09 — M4 checkpoint and human handoff

Objective: prove a genuine pause, state retention, response validation, and correct resume route.

- Configured an allowlisted in-memory LangGraph serializer and stable thread IDs.
- Added a committed waiting state followed by `interrupt`; resume uses `Command(resume=...)`.
- Verified approval, rejection, and clarification are distinct. Clarification clears derived state and re-enters retrieval; human acceptance does not rewrite a failed automated review as approved.

## 2026-10-09 — M5 interfaces and verification

Objective: supply a runnable local demonstration and record real results.

- Added CLI, JSON export, sample brief, and a Streamlit inspection/resume UI.
- Executed the installed CLI on the sample brief with `review_fail_then_pass`: observed one evidence passage, three ads, one revision, two reviews, and final `approved` status.
- Started Streamlit headlessly and received HTTP `200: ok` from its health endpoint.
- Created `.venv` under D:. Initial dependency installation was blocked by sandbox networking; the scoped approved retry installed pytest tooling successfully.
- First complete test run: 20 passed, 7 failed due to the grounding whitespace defect described above.
- Final test command: `.\.venv\Scripts\python.exe -m pytest`.
- Final acceptance result: 28 passed, 0 failed, 0 skipped in 1.40 seconds.
- No live API key was configured. The OpenAI adapter was exercised with a local structured-response stub; live inference remains unverified.

Lessons: committing the waiting state in a separate node makes interrupt inspection unambiguous; objective grounding comparisons must normalize formatting without weakening exact source support; a one-document BM25 corpus can yield non-positive scores, so relevance must consider lexical overlap rather than treating score positivity as the sole match condition.

## 2026-10-09 — expanded flow evaluation and Gemini support

Objective: add only tests that exercise meaningful workflow or provider boundaries and make the available Gemini credential path usable.

- Added a Gemini adapter using Google's documented OpenAI-compatible base URL and structured chat-completion parsing; no extra provider SDK was required.
- Added CLI/UI Gemini selection and provider-specific environment validation. Keys remain environment-only.
- Added tests for CLI success/export, noninteractive human rejection, invalid inputs, missing credentials, and terminal CLI failures.
- Added isolated supervisor, review, and revision provider-failure flows; actual step-limit exhaustion; factual-update resume; and pre-generation risky-claim routing.
- Added structured parse and connection-failure contract tests for both real-provider adapters.
- Final expanded result after local environment-loader tests: 46 passed, 0 failed, 0 skipped in 1.49 seconds.
- Coverage result: 86% overall and 99% for the LangGraph orchestration module. Streamlit event code is not represented as unit-covered; its server health is separately smoke-tested.
- Added an ignored local `.env` loader and selected free-tier `gemini-3.1-flash-lite` as the local default.
- Live Gemini authentication reached the provider. Three free-tier model attempts all returned bounded HTTP 503 high-demand responses; no successful content suite is claimed. The failure artifact is saved at `artifacts/live-gemini-result.json`.

## 2026-10-09 — vector RAG, source formats, and autonomous analysis

Objective: close the gaps against the final problem statement while leaving live Gemini validation deferred.

- Replaced BM25 as the primary retriever with cached `all-MiniLM-L6-v2` embeddings and persistent Chroma collections. BM25 remains an explicit, recorded fallback.
- Added verified semantic paraphrase retrieval and persistent-index reuse tests.
- Added real PDF parsing, Google Sheets CSV/XLSX export ingestion, multi-file loading, and Notion Markdown/CSV ZIP expansion.
- Added a structured campaign-analysis state and graph action for audience, tone, value proposition, key messages, rationale, and missing facts.
- Preserved user audience/tone overrides and passed resolved analysis into generation and review.
- Extended CLI/UI to accept multiple source formats and optional audience/tone fields; added a Campaign Analysis UI tab.
- Created a three-source sample using vector mode. The deterministic first review failed, routed through one revision, then passed. Exact output is in `examples/sample_content_suite.json`.
- Final acceptance tests: 57 passed, 0 failed, 0 skipped in 11.97 seconds.
- Final coverage: 87% overall; orchestration 97%; ingestion 91%; vector retrieval 92%.
- `pip check` exposed ambient conflicts from inherited system packages (OpenCV/SHAP expect NumPy 2; Google/gRPC packages expect older protobuf). They were not changed because the verified project uses the working installed Chroma/MiniLM stack and changing global scientific packages would be unrelated and risky.

Implementation lesson: local embedding initialization must default to cache-only in a network-restricted sandbox. Otherwise a harmless Hugging Face metadata probe can fail and trigger the correctly implemented lexical fallback even when model weights are already cached.

## 2026-10-09 — isolated environment and durable offline completion

- Created `.venv-clean` on D: without `--system-site-packages`; pinned the pip cache to the project on D:.
- Added the compatible `protobuf>=5.29,<6` intersection after the clean resolver exposed an otherwise hidden Streamlit/telemetry ambiguity.
- `pip check` now reports no broken requirements.
- Added SQLite LangGraph checkpoints, explicit connection lifecycle, stable CLI thread IDs, and restart/resume commands.
- Verified durability by closing one runner, reopening the database with a new runner, and completing its paused human-review flow.
- Added a restricted public Google Sheets connector with HTTPS/host/path validation, numeric `gid` validation, a 5 MB limit, and original-URL attribution.
- Added Streamlit support for public Sheets and persistent checkpoints.
- Added tests for persistent runner resume, CLI cross-run resume, public Sheet validation/attribution, and a complete Streamlit upload-to-approved flow.
- Final result after supervisor fallback coverage: 69 passed, 0 failed, 0 skipped; 94% instrumented core coverage and 96% orchestration coverage.
- Verified a successful live Gemini campaign from the sample PDF: vector retrieval, two evidence passages, four formats, three ads, one passing review, zero revisions, and zero errors.
