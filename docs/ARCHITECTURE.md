# Architecture

## Components

| Component | Responsibility |
|---|---|
| `models/schemas.py` | Strict inputs, analysis, evidence, content, review, action, error, human, and workflow contracts |
| `ingestion/loader.py` | PDF, text/Markdown, CSV/XLSX, multi-file, and Notion ZIP normalization |
| `ingestion/google_sheets.py` | Allowlisted public Google Sheets URL validation and bounded CSV download |
| `retrieval/vector.py` | MiniLM embeddings, persistent Chroma index, semantic retrieval, BM25 fallback |
| `retrieval/local.py` | Chunking, BM25 fallback, and narrow evidence-risk checks |
| `llm/base.py` | Provider-independent analysis, generation, review, and supervisor protocol |
| `llm/fake.py` | Deterministic offline analysis/content/review fixtures |
| `llm/gemini_client.py`, `llm/openai_client.py` | Real structured-output adapters |
| `agents/supervisor.py` | Legal-action calculation and decision validation |
| `agents/reviewer.py` | Deterministic checks plus qualitative review result |
| `orchestration/graph.py` | Conditional nodes, limits, checkpoint, interrupt, and resume |
| CLI and Streamlit | Upload, execution, inspection, human response, and export |

## Ingestion and vector RAG

Every uploaded source becomes a normalized `ProductBrief` with a stable source identifier and media type. A Notion ZIP expands into separately attributed Markdown/CSV documents. Spreadsheet rows become labelled records so dates, audiences, product names, and messages remain retrievable.

The primary retriever chunks every document, embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`, and stores them in a content-fingerprinted Chroma collection under `.gtm-vector-store`. Queries are embedded and ranked by cosine similarity. Returned evidence preserves filename, passage ID, chunk index, text, and score. A verified semantic test retrieves a release-planning passage from a paraphrased product-launch query.

If model/store initialization or querying fails, the workflow records `lexical_fallback` plus the error and uses BM25 rather than silently calling lexical retrieval “semantic.”

## Campaign analysis

After retrieval, a separate structured analysis action derives target audience, tone, value proposition, key messages, and missing facts from approved evidence. Supplied audience/tone values are constraints and are never overwritten. The resolved campaign and analysis are passed into generation and review.

## Stateful routing

The supervisor can select only actions legal for current state:

1. retrieve approved evidence;
2. pause when evidence is absent, contradictory, or risky;
3. analyze campaign positioning;
4. generate all formats;
5. review generated assets;
6. revise a failed review while attempts remain;
7. request a human after unresolved limits;
8. complete only after review approval or an explicit labelled human override;
9. fail after terminal error or rejection.

Pydantic constrains action names and `legal_actions()` enforces ordering. Provider/malformed outputs retry twice, revisions default to two, and supervisor steps default to twenty.

## Grounding and review

Every asset must carry retrieved source IDs, and each verified fact must occur in normalized evidence. Deterministic checks cover all required formats, email/blog headings, three distinct ads, source attribution, and fact grounding. Qualitative review covers tone, audience, consistency, missing information, and unsupported claims. Objective-check failure always forces rejection regardless of model opinion.

## Human handoff

`prepare_human` commits `waiting_human`; the next node executes LangGraph `interrupt`. `Command(resume=...)` distinguishes approval, rejection, and clarification. Clarification appends approved factual context, clears derived evidence/analysis/content/review, and re-enters retrieval. Human acceptance of a failed automated review remains labelled `unapproved`.

The default library checkpointer remains in-memory for isolated tests. Passing `checkpoint_path` selects LangGraph's SQLite saver with the same allowlisted serializer. The CLI exposes checkpoint database, thread ID, and resume arguments; tests close the first runner, open a second runner against the same database, reload the waiting state, and resume it.

## Connector boundary

Public Google Sheets use a restricted HTTPS `docs.google.com/spreadsheets` connector with a 5 MB response limit and original-URL attribution. CSV/XLSX and Notion ZIP exports require no account access. Private Sheets and direct Notion workspaces require OAuth or service credentials plus explicit user authorization, so those connectors are deliberately outside the current implementation rather than simulated.
