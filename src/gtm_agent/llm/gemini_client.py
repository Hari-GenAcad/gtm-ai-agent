from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from gtm_agent.llm.base import MalformedModelOutput, ProviderFailure
from gtm_agent.llm.openai_client import OpenAILLM


T = TypeVar("T", bound=BaseModel)


class GeminiLLM(OpenAILLM):
    """Gemini structured-output adapter through Google's OpenAI-compatible API."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, *, model: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.client = client or OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=self.BASE_URL,
            timeout=45.0,
            max_retries=0,
        )

    def _parse(self, *, system: str, payload: dict[str, Any], schema: type[T]) -> T:
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format=schema,
            )
            parsed = completion.choices[0].message.parsed
        except OpenAIError as exc:
            raise ProviderFailure(str(exc)) from exc
        except (TypeError, ValueError, ValidationError, AttributeError, IndexError) as exc:
            raise MalformedModelOutput(str(exc)) from exc
        if not isinstance(parsed, schema):
            raise MalformedModelOutput(f"provider did not return {schema.__name__}")
        return parsed
