from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.documents import Document

from langchain_vectoramp import VectorAmpVectorStore


def json_response(data: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=data, headers={"content-type": "application/json"})


def test_add_texts_uses_vectoramp_hosted_embedding_and_insert() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        calls.append((request.method, request.url.path, body))
        assert request.headers["x-api-key"] == "test-key"
        if request.url.path == "/datasets/ds_1/embed":
            assert body == {"texts": ["hello", "world"]}
            return json_response({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
        if request.url.path == "/datasets/ds_1/insert":
            return json_response({"inserted": 2})
        raise AssertionError(str(request.url))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    store = VectorAmpVectorStore(
        api_key="test-key",
        base_url="https://api.test",
        dataset_id="ds_1",
        client=None,
        timeout=1,
    )
    # Inject the mock after construction so constructor behavior stays simple.
    store._client._client = http_client  # type: ignore[attr-defined]

    ids = store.add_texts(
        ["hello", "world"], ids=["a", "b"], metadatas=[{"kind": "greeting"}, {"kind": "noun"}]
    )

    assert ids == ["a", "b"]
    insert_body = calls[1][2]
    assert insert_body == {
        "vectors": [
            {"id": "a", "values": [0.1, 0.2], "metadata": {"kind": "greeting", "text": "hello"}},
            {"id": "b", "values": [0.3, 0.4], "metadata": {"kind": "noun", "text": "world"}},
        ]
    }


def test_add_texts_preserves_integer_ids_as_json_numbers() -> None:
    insert_body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/datasets/ds_1/embed":
            return json_response({"embeddings": [[0.1], [0.2]]})
        if request.url.path == "/datasets/ds_1/insert":
            insert_body.update(json.loads(request.content))
            return json_response({"inserted": 2})
        raise AssertionError(str(request.url))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    store = VectorAmpVectorStore(api_key="test-key", base_url="https://api.test", dataset_id="ds_1")
    store._client._client = http_client  # type: ignore[attr-defined]

    returned = store.add_texts(["a", "b"], ids=[1, 2])

    # Ids returned to the caller keep their integer type...
    assert returned == [1, 2]
    assert all(isinstance(value, int) for value in returned)
    # ...and the raw request bytes carry JSON numbers, not quoted strings.
    raw = json.dumps(insert_body)
    assert '"id": 1' in raw
    assert '"id": 2' in raw
    assert '"id": "1"' not in raw
    assert insert_body["vectors"][0]["id"] == 1
    assert insert_body["vectors"][1]["id"] == 2


def test_add_texts_keeps_string_ids_as_strings() -> None:
    insert_body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/datasets/ds_1/embed":
            return json_response({"embeddings": [[0.1]]})
        if request.url.path == "/datasets/ds_1/insert":
            insert_body.update(json.loads(request.content))
            return json_response({"inserted": 1})
        raise AssertionError(str(request.url))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    store = VectorAmpVectorStore(api_key="test-key", base_url="https://api.test", dataset_id="ds_1")
    store._client._client = http_client  # type: ignore[attr-defined]

    returned = store.add_texts(["a"], ids=["doc-a"])
    assert returned == ["doc-a"]
    assert insert_body["vectors"][0]["id"] == "doc-a"


def test_similarity_search_expands_rerank_true_to_full_object() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return json_response({"results": []})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    store = VectorAmpVectorStore(api_key="test-key", base_url="https://api.test", dataset_id="ds_1")
    store._client._client = http_client  # type: ignore[attr-defined]

    store.similarity_search("q", k=2, rerank=True)

    assert seen["body"]["rerank"] == {
        "enabled": True,
        "provider": "vectoramp",
        "model": "VectorAmp-Rerank-v1",
    }


def test_similarity_search_filters_alias_matches_filter() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return json_response({"results": []})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    store = VectorAmpVectorStore(api_key="test-key", base_url="https://api.test", dataset_id="ds_1")
    store._client._client = http_client  # type: ignore[attr-defined]

    # The ``filters`` alias maps to the same API ``filters`` field as ``filter``.
    store.similarity_search("q", k=1, filters={"tenant": "acme"})
    assert seen["body"]["filters"] == {"tenant": "acme"}


def test_retry_job_posts_to_ingestion_retry_endpoint() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return json_response({"job_id": "job_2", "status": "pending"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    store = VectorAmpVectorStore(api_key="test-key", dataset_id="ds_1")
    store._client._client = http_client  # type: ignore[attr-defined]
    store._client.base_url = "https://api.test"  # type: ignore[attr-defined]

    assert store._client.retry_job("job_1") == {"job_id": "job_2", "status": "pending"}
    assert seen == {"method": "POST", "path": "/ingestion/jobs/job_1/retry"}


def test_similarity_search_with_filter_and_scores() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return json_response(
            {
                "results": [
                    {
                        "id": "v1",
                        "score": 0.91,
                        "doc_value": "VectorAmp is fast",
                        "metadata": {"source": "docs", "text": "stored text"},
                    }
                ]
            }
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    store = VectorAmpVectorStore(api_key="test-key", dataset_id="ds_1")
    store._client._client = http_client  # type: ignore[attr-defined]
    store._client.base_url = "https://api.test"  # type: ignore[attr-defined]

    results = store.similarity_search_with_score(
        "what is vectoramp?",
        k=3,
        filter={"source": "docs"},
        include_metadata=True,
        hybrid=True,
        rerank={"enabled": True},
    )

    assert seen["body"] == {
        "query_text": "what is vectoramp?",
        "top_k": 3,
        "include_documents": True,
        "filters": {"source": "docs"},
        "include_metadata": True,
        "hybrid": True,
        "rerank": {"enabled": True},
    }
    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "VectorAmp is fast"
    assert doc.metadata["source"] == "docs"
    assert doc.metadata["id"] == "v1"
    assert doc.id == "v1"
    assert score == 0.91


def test_similarity_search_rejects_separate_search_text_knob() -> None:
    store = VectorAmpVectorStore(api_key="test-key", dataset_id="ds_1")

    with pytest.raises(ValueError, match="LangChain query argument"):
        store.similarity_search("primary text", search_text="different text")



def test_from_documents_and_retriever_with_mock_client() -> None:
    class MockDatasets:
        def __init__(self) -> None:
            self.add_calls: list[dict[str, Any]] = []

        def add_texts(self, dataset_id: str, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            self.add_calls.append({"dataset_id": dataset_id, "texts": texts, **kwargs})
            return {"ids": kwargs["ids"]}

        def search(self, dataset_id: str, query: str, **kwargs: Any) -> dict[str, Any]:
            assert dataset_id == "ds_1"
            assert kwargs["filters"] == {"tenant": "acme"}
            return {"results": [{"vector_id": "v1", "score": 1.0, "metadata": {"text": query}}]}

    class MockClient:
        def __init__(self) -> None:
            self.datasets = MockDatasets()

    client = MockClient()
    docs = [Document(page_content="alpha", metadata={"tenant": "acme"}, id="doc-1")]

    store = VectorAmpVectorStore.from_documents(docs, client=client, dataset_id="ds_1")

    assert client.datasets.add_calls == [
        {
            "dataset_id": "ds_1",
            "texts": ["alpha"],
            "ids": ["doc-1"],
            "metadatas": [{"tenant": "acme"}],
        }
    ]
    retriever = store.as_retriever(search_kwargs={"k": 1, "filter": {"tenant": "acme"}})
    assert retriever.invoke("alpha")[0].page_content == "alpha"


@pytest.mark.asyncio
async def test_async_add_and_search() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        if request.url.path == "/datasets/ds_1/embed":
            return json_response({"embeddings": [[0.1]]})
        if request.url.path == "/datasets/ds_1/insert":
            assert body["vectors"][0]["id"] == "async-id"
            return json_response({"inserted": 1})
        if request.url.path == "/datasets/ds_1/search":
            assert body["filters"] == {"kind": "async"}
            return json_response(
                {"results": [{"id": "v1", "score": 0.5, "doc_value": "async doc"}]}
            )
        raise AssertionError(str(request.url))

    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.test"
    )
    store = VectorAmpVectorStore(api_key="test-key", base_url="https://api.test", dataset_id="ds_1")
    store._client._async_client = async_client  # type: ignore[attr-defined]

    assert await store.aadd_texts(["async"], ids=["async-id"]) == ["async-id"]
    docs = await store.asimilarity_search("async", k=1, filter={"kind": "async"})
    assert docs == [Document(page_content="async doc", metadata={"id": "v1"}, id="v1")]
    await async_client.aclose()


def test_minimal_init_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Minimal init: only the dataset is named; the key comes from the environment.
    monkeypatch.setenv("VECTORAMP_API_KEY", "env-key")
    store = VectorAmpVectorStore(dataset_id="ds_1")
    assert store._client.api_key == "env-key"  # type: ignore[attr-defined]
    assert store._client.base_url == "https://api.vectoramp.com"  # type: ignore[attr-defined]


def test_embedding_argument_is_accepted_but_ignored() -> None:
    class SentinelEmbeddings:
        # Not a real Embeddings impl; it must never be called for add/search.
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("VectorAmp embeds server-side; embedding must be ignored.")

        def embed_query(self, text: str) -> list[float]:
            raise AssertionError("VectorAmp embeds server-side; embedding must be ignored.")

    sentinel = SentinelEmbeddings()
    store = VectorAmpVectorStore(
        api_key="test-key", dataset_id="ds_1", embedding=sentinel  # type: ignore[arg-type]
    )
    # Exposed via the LangChain ``embeddings`` property, but never invoked.
    assert store.embeddings is sentinel


def test_dataset_name_resolution() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/datasets"
        assert dict(request.url.params) == {"limit": "100", "offset": "0"}
        return json_response({"datasets": [{"id": "ds_1", "name": "docs"}]})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    from langchain_vectoramp.client import VectorAmpHTTPClient

    client = VectorAmpHTTPClient(
        api_key="test-key", base_url="https://api.test", http_client=http_client
    )
    store = VectorAmpVectorStore(client=client, dataset_name="docs")

    assert store.dataset_id == "ds_1"


def test_loader_loads_search_results_and_merges_metadata() -> None:
    from langchain_vectoramp import VectorAmpLoader

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return json_response(
            {
                "results": [
                    {
                        "id": "v1",
                        "score": 0.8,
                        "doc_value": "loaded doc",
                        "metadata": {"source": "api"},
                    }
                ]
            }
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    from langchain_vectoramp.client import VectorAmpHTTPClient

    client = VectorAmpHTTPClient(
        api_key="test-key", base_url="https://api.test", http_client=http_client
    )
    loader = VectorAmpLoader(
        client=client,
        dataset_id="ds_1",
        query="load docs",
        filter={"source": "api"},
        k=2,
        metadata={"loaded_by": "vectoramp"},
    )

    docs = loader.load()

    assert seen["body"] == {
        "query_text": "load docs",
        "top_k": 2,
        "include_documents": True,
        "filters": {"source": "api"},
        "include_metadata": True,
    }
    assert docs == [
        Document(
            page_content="loaded doc",
            metadata={"loaded_by": "vectoramp", "source": "api", "id": "v1"},
            id="v1",
        )
    ]


def test_loader_lists_dataset_documents_without_query() -> None:
    from langchain_vectoramp import VectorAmpLoader

    class MockClient:
        def list_documents(
            self,
            dataset_id: str,
            *,
            limit: int,
            cursor: str | None,
            status: str | None,
        ) -> dict[str, Any]:
            assert dataset_id == "ds_1"
            assert limit == 2
            assert cursor is None
            assert status == "ready"
            return {
                "documents": [
                    {"id": "doc_1", "file_name": "a.md", "download_available": True}
                ],
                "next_cursor": None,
            }

        def download_document(self, dataset_id: str, document_id: str) -> bytes:
            assert dataset_id == "ds_1"
            assert document_id == "doc_1"
            return b"full doc"

    loader = VectorAmpLoader(
        client=MockClient(),
        dataset_id="ds_1",
        k=2,
        metadata={"loaded_by": "vectoramp"},
    )
    assert loader.load() == [
        Document(
            page_content="full doc",
            metadata={
                "loaded_by": "vectoramp",
                "id": "doc_1",
                "file_name": "a.md",
                "download_available": True,
            },
            id="doc_1",
        )
    ]


def test_http_client_download_document_follows_redirects() -> None:
    from langchain_vectoramp.client import VectorAmpHTTPClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/datasets/ds_1/documents/doc_1/download":
            return httpx.Response(307, headers={"location": "https://download.test/doc_1"})
        if str(request.url) == "https://download.test/doc_1":
            return httpx.Response(200, content=b"full doc")
        raise AssertionError(str(request.url))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = VectorAmpHTTPClient(
        api_key="test-key", base_url="https://api.test", http_client=http_client
    )

    assert client.download_document("ds_1", "doc_1") == b"full doc"
