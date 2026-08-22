from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_vectoramp import VectorAmpIntelligence


def _intel_with_capture(calls: list[dict[str, Any]], **kwargs: Any):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        calls.append({"method": request.method, "path": request.url.path, "body": body})
        if request.url.path == "/datasets":
            return httpx.Response(
                200,
                json={"datasets": [{"id": "ds_named", "name": "Docs"}]},
                headers={"content-type": "application/json"},
            )
        return httpx.Response(
            200,
            json={"answer": "the answer", "sources": [{"title": "doc"}]},
            headers={"content-type": "application/json"},
        )

    intel = VectorAmpIntelligence(api_key="test-key", base_url="https://api.test", **kwargs)
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
    # "Every dataset I can see" is spelled by omitting the scope. The retired
    # dataset_id -- and its "all" sentinel -- now draw a 400 from the API.
    assert "dataset_ids" not in body
    assert "dataset_id" not in body
    assert body["stream"] is False
    assert body["include_sources"] is True
    assert "conversation_history" not in body


def test_ask_scopes_to_every_configured_dataset() -> None:
    calls: list[dict[str, Any]] = []
    intel = _intel_with_capture(calls, dataset_ids=["ds_1", "ds_2", "ds_3"])

    intel.ask("Which contracts renew in Q4?")

    assert calls[0]["body"]["dataset_ids"] == ["ds_1", "ds_2", "ds_3"]


def test_singular_and_plural_dataset_ids_merge_into_one_scope() -> None:
    calls: list[dict[str, Any]] = []
    intel = _intel_with_capture(calls, dataset_id="ds_1", dataset_ids=["ds_2"])

    intel.ask("why?")

    assert calls[0]["body"]["dataset_ids"] == ["ds_1", "ds_2"]


def test_dataset_names_resolve_into_the_scope() -> None:
    calls: list[dict[str, Any]] = []
    intel = _intel_with_capture(calls, dataset_names=["Docs"])

    intel.ask("why?")

    assert calls[0]["path"] == "/datasets"
    assert calls[1]["body"]["dataset_ids"] == ["ds_named"]


@pytest.mark.parametrize("scope", [None, [], ["all"], ["", "  "]])
def test_empty_and_all_scopes_omit_dataset_ids(scope: Any) -> None:
    # dataset_ids stays optional; an empty scope is omitted rather than sent as
    # [], which would be a narrower, different request.
    calls: list[dict[str, Any]] = []
    intel = _intel_with_capture(calls, dataset_ids=scope)

    intel.ask("why?")

    assert "dataset_ids" not in calls[0]["body"]
    assert "dataset_id" not in calls[0]["body"]


def test_per_call_dataset_ids_override_the_configured_scope() -> None:
    calls: list[dict[str, Any]] = []
    intel = _intel_with_capture(calls, dataset_ids=["ds_1"])

    intel.ask("why?", dataset_ids=["ds_2", "ds_3"])

    assert calls[0]["body"]["dataset_ids"] == ["ds_2", "ds_3"]


def test_retired_dataset_id_override_is_refused() -> None:
    calls: list[dict[str, Any]] = []
    intel = _intel_with_capture(calls)

    with pytest.raises(TypeError, match="dataset_ids"):
        intel.ask("why?", dataset_id="ds_1")

    assert calls == []


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
    # Minimal init: no api_key argument, no dataset -> env key + every visible dataset,
    # which the API spells as an empty (and therefore omitted) scope.
    monkeypatch.setenv("VECTORAMP_API_KEY", "env-key")
    intel = VectorAmpIntelligence()
    assert intel._client.api_key == "env-key"  # type: ignore[attr-defined]
    assert intel._datasets() == []  # type: ignore[attr-defined]
