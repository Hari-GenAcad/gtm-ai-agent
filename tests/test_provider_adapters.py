import httpx
import pytest
from openai import APIConnectionError

from gtm_agent.llm import GeminiLLM, OpenAILLM
from gtm_agent.llm.base import MalformedModelOutput, ProviderFailure
from gtm_agent.llm.fake import FakeLLM
from gtm_agent.models.schemas import EvidencePassage


def evidence() -> list[EvidencePassage]:
    return [
        EvidencePassage(
            passage_id="brief.md#chunk-1",
            source_id="brief.md",
            text="LaunchPad helps revenue teams launch campaigns.",
            score=1,
            chunk_index=0,
        )
    ]


class ParsedResponse:
    def __init__(self, parsed):
        self.output_parsed = parsed


class GeminiCompletion:
    def __init__(self, parsed):
        message = type("Message", (), {"parsed": parsed})()
        self.choices = [type("Choice", (), {"message": message})()]


def test_gemini_adapter_uses_compatible_structured_parse(brief, campaign) -> None:
    expected = FakeLLM().generate(brief, campaign, evidence())

    class Completions:
        def parse(self, **kwargs):
            assert kwargs["model"] == "gemini-test"
            assert kwargs["response_format"].__name__ == "ContentSuite"
            assert kwargs["messages"][0]["role"] == "system"
            return GeminiCompletion(expected)

    client = type(
        "Client",
        (),
        {"beta": type("Beta", (), {"chat": type("Chat", (), {"completions": Completions()})()})()},
    )()
    adapter = GeminiLLM(model="gemini-test", client=client)
    assert adapter.generate(brief, campaign, evidence()) == expected


@pytest.mark.parametrize("adapter_class", [OpenAILLM, GeminiLLM])
def test_adapters_classify_missing_parsed_output(adapter_class, brief, campaign) -> None:
    if adapter_class is OpenAILLM:
        parser = type("Responses", (), {"parse": lambda self, **kwargs: ParsedResponse(None)})()
        client = type("Client", (), {"responses": parser})()
    else:
        parser = type("Completions", (), {"parse": lambda self, **kwargs: GeminiCompletion(None)})()
        client = type(
            "Client",
            (),
            {"beta": type("Beta", (), {"chat": type("Chat", (), {"completions": parser})()})()},
        )()
    with pytest.raises(MalformedModelOutput, match="did not return ContentSuite"):
        adapter_class(model="test", client=client).generate(brief, campaign, evidence())


@pytest.mark.parametrize("adapter_class", [OpenAILLM, GeminiLLM])
def test_adapters_classify_provider_connection_failure(adapter_class, brief, campaign) -> None:
    def fail(**kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "https://provider.invalid"))

    parser = type("Parser", (), {"parse": lambda self, **kwargs: fail(**kwargs)})()
    if adapter_class is OpenAILLM:
        client = type("Client", (), {"responses": parser})()
    else:
        client = type(
            "Client",
            (),
            {"beta": type("Beta", (), {"chat": type("Chat", (), {"completions": parser})()})()},
        )()
    with pytest.raises(ProviderFailure):
        adapter_class(model="test", client=client).generate(brief, campaign, evidence())

