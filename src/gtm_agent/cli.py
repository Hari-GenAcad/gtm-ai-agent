from __future__ import annotations

import argparse
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gtm_agent.config import load_local_env
from gtm_agent.ingestion import load_documents, load_public_google_sheet, merge_documents
from gtm_agent.llm import FakeLLM, FakeScenario, GeminiLLM, OpenAILLM
from gtm_agent.models.schemas import CampaignRequest, HumanReviewResponse, WorkflowStatus
from gtm_agent.orchestration import WorkflowRunner


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def state_json(state: dict[str, Any]) -> str:
    return json.dumps(state, default=_json_default, indent=2, ensure_ascii=False)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the evidence-grounded GTM content workflow.")
    result.add_argument(
        "brief", nargs="*", type=Path, help="One or more .txt, .md, .pdf, .csv, .xlsx, or Notion .zip files"
    )
    result.add_argument("--google-sheet", action="append", default=[], help="Public Google Sheets URL (repeatable)")
    result.add_argument("--objective", default="")
    result.add_argument("--audience", default="", help="Optional override; inferred when omitted")
    result.add_argument("--tone", default="", help="Optional override; inferred when omitted")
    result.add_argument("--provider", choices=["fake", "gemini", "openai"], default="fake")
    result.add_argument("--model", help="Provider model override; otherwise its environment/default is used")
    result.add_argument("--scenario", choices=[item.value for item in FakeScenario], default="happy")
    result.add_argument("--max-revisions", type=int, default=2)
    result.add_argument("--human-response", choices=["approve", "reject"], help="Non-interactive pause response")
    result.add_argument("--checkpoint-db", type=Path, help="SQLite checkpoint database for restart-safe runs")
    result.add_argument("--thread-id", help="Stable thread id for a new checkpointed run")
    result.add_argument("--resume-thread", help="Resume a paused thread from --checkpoint-db")
    result.add_argument("--output", type=Path, help="Optional JSON export path")
    return result


def _human_response(args: argparse.Namespace, request: BaseModel) -> HumanReviewResponse:
    print("\nHuman review required:")
    print(json.dumps(request.model_dump(mode="json"), indent=2))
    decision = args.human_response
    if decision is None:
        decision = input("Decision [approve/reject/clarify]: ").strip().lower()
    if decision == "clarify":
        comment = input("Clarification: ").strip()
        return HumanReviewResponse(decision="clarify", comment=comment)
    comment = "Explicit CLI response."
    return HumanReviewResponse(decision=decision, comment=comment)


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = parser().parse_args(argv)
    if args.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is required for --provider openai.")
            return 2
        llm = OpenAILLM(model=args.model)
    elif args.provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            print("GEMINI_API_KEY is required for --provider gemini.")
            return 2
        llm = GeminiLLM(model=args.model)
    else:
        llm = FakeLLM(FakeScenario(args.scenario))

    if args.resume_thread and (not args.checkpoint_db or not args.human_response):
        print("--resume-thread requires --checkpoint-db and --human-response.")
        return 2

    runner = WorkflowRunner(llm, checkpoint_path=args.checkpoint_db)
    if args.resume_thread:
        thread_id = args.resume_thread
        try:
            state = runner.resume(
                thread_id,
                HumanReviewResponse(decision=args.human_response, comment="Explicit CLI resume response."),
            )
        except Exception as exc:
            runner.close()
            print(f"Unable to resume workflow: {exc}")
            return 2
    else:
        if not args.objective.strip():
            runner.close()
            print("--objective is required for a new run.")
            return 2
        try:
            documents = load_documents(args.brief) if args.brief else []
            for url in args.google_sheet:
                documents.extend(load_public_google_sheet(url))
            brief = merge_documents(documents)
        except (FileNotFoundError, ValueError, RuntimeError, UnicodeError) as exc:
            runner.close()
            print(f"Unable to load brief: {exc}")
            return 2
        campaign = CampaignRequest(objective=args.objective, audience=args.audience, tone=args.tone)
        thread_id, state = runner.start(
            brief,
            campaign,
            documents=documents,
            thread_id=args.thread_id,
            max_revisions=max(0, args.max_revisions),
        )

    while state["status"] == WorkflowStatus.WAITING_HUMAN:
        response = _human_response(args, state["human_request"])
        state = runner.resume(thread_id, response)
        if args.human_response is not None:
            break

    rendered = state_json(state)
    print(rendered)
    if args.output:
        args.output.resolve().write_text(rendered + "\n", encoding="utf-8")
        print(f"Exported: {args.output.resolve()}")
    if args.checkpoint_db:
        print(f"Checkpoint thread: {thread_id}")
    runner.close()
    return 0 if state["status"] in {WorkflowStatus.APPROVED, WorkflowStatus.UNAPPROVED} else 1


if __name__ == "__main__":
    raise SystemExit(main())
