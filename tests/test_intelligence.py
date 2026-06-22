from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_vectoramp import VectorAmpIntelligence


def _intel_with_capture(calls: list[dict[str, Any]]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        calls.append({"method": request.method, "path": request.url.path, "body": body})
        return httpx.Response(
            200,
            json={"answer": "the answer", "sources": [{"title": "doc"}]},
            headers={"content-type": "application/json"},
        )

    intel = VectorAmpIntelligence(api_key="test-key", base_url="https://api.test")
    intel._client._client = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[attr-defined]
    return intel


def test_ask_uses_sane_defaults_with_only_a_question() -> None:
    calls: list[dict[str, Any]] = []
    intel = _intel_with_capture(calls)

    answer = intel.ask("What changed in Q4 planning?")

    assert answer == "the answer"
    assert calls[0]["path"] == "/intelligence/query"
    body = calls[0]["body"]
    # Minimal-input defaults: all datasets, non-streaming, sources on, no required extras.
    assert body["query"] == "What changed in Q4 planning?"
    assert body["dataset_id"] == "all"
    assert body["stream"] is False
    assert body["include_sources"] is True
    assert "conversation_history" not in body


def test_ask_maps_history_messages_to_conversation_history() -> None:
    calls: list[dict[str, Any]] = []
    intel = _intel_with_capture(calls)

    history = [HumanMessage("What is VectorAmp?"), AIMessage("A vector database platform.")]
    intel.ask("Does it support hybrid search?", history=history)

    body = calls[0]["body"]
    assert body["query"] == "Does it support hybrid search?"
    assert body["conversation_history"] == [
        {"role": "user", "content": "What is VectorAmp?"},
        {"role": "assistant", "content": "A vector database platform."},
    ]


def test_invoke_with_message_list_splits_last_message_as_query() -> None:
    calls: list[dict[str, Any]] = []
    intel = _intel_with_capture(calls)

    answer = intel.invoke(
        [HumanMessage("hi"), AIMessage("hello"), HumanMessage("and what about reranking?")]
    )

    assert answer == "the answer"
    body = calls[0]["body"]
    assert body["query"] == "and what about reranking?"
    assert body["conversation_history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_ask_with_sources_returns_full_payload() -> None:
    calls: list[dict[str, Any]] = []
    intel = _intel_with_capture(calls)

    result = intel.ask_with_sources("anything", top_k=7)

    assert result["sources"] == [{"title": "doc"}]
    assert calls[0]["body"]["top_k"] == 7


def test_intelligence_minimal_init_reads_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Minimal init: no api_key argument, no dataset -> env key + "all" datasets.
    monkeypatch.setenv("VECTORAMP_API_KEY", "env-key")
    intel = VectorAmpIntelligence()
    assert intel._client.api_key == "env-key"  # type: ignore[attr-defined]
    assert intel._dataset() == "all"  # type: ignore[attr-defined]
