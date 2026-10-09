# Five-Minute Demo Script

## 1. Show the three sources

Open:

- `examples/product_brief.md`
- `examples/launch_calendar.csv`
- `examples/past_campaign.md`

Explain that they represent product facts, Google Sheets calendar export, and approved past messaging.

## 2. Run vector RAG and automatic campaign analysis

```powershell
.\.venv-clean\Scripts\gtm-agent.exe `
  examples\product_brief.md `
  examples\launch_calendar.csv `
  examples\past_campaign.md `
  --objective "Create a coordinated launch campaign for LaunchPad" `
  --scenario review_fail_then_pass `
  --output demo-result.json
```

Audience and tone are omitted intentionally. Show:

- `retrieval_mode: vector`;
- evidence from all three source IDs;
- inferred audience and tone;
- analysis value proposition/key messages;
- LinkedIn, email, blog, and three ads;
- failed first review, revision, passed second review;
- final `approved` status and complete action history.

## 3. Show safety routing and human resume

Run `missing_evidence` or `review_always_fails`. Demonstrate the committed waiting state, then reject, approve with an explicit unapproved label, or clarify and resume retrieval.

## 4. Show Streamlit

```powershell
.\.venv-clean\Scripts\python.exe -m streamlit run src\gtm_agent\ui\streamlit_app.py
```

Upload all three sample files. Show Evidence, Campaign Analysis, Content, Review, and Action History tabs plus export.

## 5. Show verification

```powershell
.\.venv-clean\Scripts\python.exe -m pytest
.\.venv-clean\Scripts\python.exe -m pytest --cov=gtm_agent --cov-report=term-missing
```

Show 69 passing tests, 94% instrumented core coverage, the clean dependency check, the Streamlit acceptance test, and the successful live Gemini run. Explain that the deterministic provider validates repeatable workflow paths while Gemini demonstrates the final writing quality.
