"""Small internal VectorAmp HTTP client used by the LangChain adapter."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import httpx

JSON = dict[str, Any]
FilterValue = Union[str, int, float, bool, None]
Filters = Mapping[str, FilterValue]


class VectorAmpClientError(RuntimeError):
    """Raised when the VectorAmp API returns an error."""


class VectorAmpHTTPClient:
    """Minimal client for dataset text insertion and text search endpoints."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.vectoramp.com",
        timeout: float = 30.0,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("VECTORAMP_API_KEY") or ""
        if not self.api_key:
            raise ValueError("api_key is required or VECTORAMP_API_KEY must be set.")
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._owns_async_client = async_http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)
        self._async_client = async_http_client or httpx.AsyncClient(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    async def aclose(self) -> None:
        if self._owns_async_client:
            await self._async_client.aclose()

    def get_dataset_id(self, *, dataset_id: Optional[str], dataset_name: Optional[str]) -> str:
        if dataset_id:
            return dataset_id
        if not dataset_name:
            raise ValueError("dataset_id or dataset_name is required.")
        page = self._request("GET", "/datasets", params={"limit": 100, "offset": 0})
        for dataset in page.get("datasets", []):
            if dataset.get("name") == dataset_name:
                value = dataset.get("id") or dataset.get("dataset_id")
                if value is not None:
                    return str(value)
        raise ValueError(f"Dataset named {dataset_name!r} was not found.")

    async def aget_dataset_id(
        self, *, dataset_id: Optional[str], dataset_name: Optional[str]
    ) -> str:
        if dataset_id:
            return dataset_id
        if not dataset_name:
            raise ValueError("dataset_id or dataset_name is required.")
        page = await self._arequest("GET", "/datasets", params={"limit": 100, "offset": 0})
        for dataset in page.get("datasets", []):
            if dataset.get("name") == dataset_name:
                value = dataset.get("id") or dataset.get("dataset_id")
                if value is not None:
                    return str(value)
        raise ValueError(f"Dataset named {dataset_name!r} was not found.")

    def add_texts(
        self,
        dataset_id: str,
        texts: Sequence[str],
        *,
        ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> list[str]:
        embeddings = self._embed(dataset_id, texts)
        vector_ids = self._build_ids(texts, ids)
        self._insert_vectors(dataset_id, texts, embeddings, vector_ids, metadatas)
        return vector_ids

    async def aadd_texts(
        self,
        dataset_id: str,
        texts: Sequence[str],
        *,
        ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> list[str]:
        embeddings = await self._aembed(dataset_id, texts)
        vector_ids = self._build_ids(texts, ids)
        await self._ainsert_vectors(dataset_id, texts, embeddings, vector_ids, metadatas)
        return vector_ids

    def search(self, dataset_id: str, query: str, *, k: int, **kwargs: Any) -> JSON:
        return self._request(
            "POST", f"/datasets/{dataset_id}/search", json=self._search_body(query, k, kwargs)
        )

    def intelligence_query(self, body: Mapping[str, Any]) -> JSON:
        """POST an Intelligence/RAG query and return the JSON answer."""
        return self._request("POST", "/intelligence/query", json=dict(body))

    async def aintelligence_query(self, body: Mapping[str, Any]) -> JSON:
        """Async variant of :meth:`intelligence_query`."""
        return await self._arequest("POST", "/intelligence/query", json=dict(body))

    def list_documents(
        self,
        dataset_id: str,
        *,
        limit: int = 50,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
    ) -> JSON:
        return self._request(
            "GET",
            f"/datasets/{dataset_id}/documents",
            params={"limit": limit, "cursor": cursor, "status": status},
        )

    def retry_job(self, job_id: str) -> JSON:
        """Queue a fresh full-rerun job from an eligible failed or cancelled job."""
        return self._request("POST", f"/ingestion/jobs/{job_id}/retry")

    async def aretry_job(self, job_id: str) -> JSON:
        """Async variant of :meth:`retry_job`."""
        return await self._arequest("POST", f"/ingestion/jobs/{job_id}/retry")

    def download_document(self, dataset_id: str, document_id: str) -> bytes:
        response = self._client.request(
            "GET",
            f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/download",
            headers=self._headers(),
            follow_redirects=True,
        )
        if response.status_code >= 300:
            raise VectorAmpClientError(
                f"VectorAmp API error {response.status_code}: {response.text}"
            )
        return response.content

    async def asearch(self, dataset_id: str, query: str, *, k: int, **kwargs: Any) -> JSON:
        return await self._arequest(
            "POST", f"/datasets/{dataset_id}/search", json=self._search_body(query, k, kwargs)
        )

    def _embed(self, dataset_id: str, texts: Sequence[str]) -> list[Sequence[float]]:
        response = self._request(
            "POST", f"/datasets/{dataset_id}/embed", json={"texts": list(texts)}
        )
        return self._extract_embeddings(response, len(texts))

    async def _aembed(self, dataset_id: str, texts: Sequence[str]) -> list[Sequence[float]]:
        response = await self._arequest(
            "POST", f"/datasets/{dataset_id}/embed", json={"texts": list(texts)}
        )
        return self._extract_embeddings(response, len(texts))

    def _insert_vectors(
        self,
        dataset_id: str,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
        metadatas: Optional[Sequence[Mapping[str, Any]]],
    ) -> JSON:
        return self._request(
            "POST",
            f"/datasets/{dataset_id}/insert",
            json={"vectors": self._vectors(texts, embeddings, ids, metadatas)},
        )

    async def _ainsert_vectors(
        self,
        dataset_id: str,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
        metadatas: Optional[Sequence[Mapping[str, Any]]],
    ) -> JSON:
        return await self._arequest(
            "POST",
            f"/datasets/{dataset_id}/insert",
            json={"vectors": self._vectors(texts, embeddings, ids, metadatas)},
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        response = self._client.request(
            method,
            f"{self.base_url}{path}",
            headers=self._headers(),
            params=params,
            json=json,
        )
        return self._decode_response(response)

    async def _arequest(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        response = await self._async_client.request(
            method,
            f"{self.base_url}{path}",
            headers=self._headers(),
            params=params,
            json=json,
        )
        return self._decode_response(response)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _decode_response(response: httpx.Response) -> JSON:
        if response.status_code >= 400:
            raise VectorAmpClientError(
                f"VectorAmp API error {response.status_code}: {response.text}"
            )
        if response.status_code == 204 or not response.content:
            return {}
        data = response.json()
        if not isinstance(data, dict):
            raise VectorAmpClientError("VectorAmp API returned a non-object JSON response.")
        return data

    @staticmethod
    def _extract_embeddings(response: Mapping[str, Any], count: int) -> list[Sequence[float]]:
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != count:
            raise VectorAmpClientError("Embedding response length did not match input texts.")
        return embeddings

    @staticmethod
    def _build_ids(texts: Sequence[str], ids: Optional[Sequence[str]]) -> list[str]:
        if ids is not None:
            if len(ids) != len(texts):
                raise ValueError("ids length must match texts length.")
            return [str(value) for value in ids]
        return [str(uuid.uuid4()) for _ in texts]

    @staticmethod
    def _vectors(
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
        metadatas: Optional[Sequence[Mapping[str, Any]]],
    ) -> list[JSON]:
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length.")
        vectors: list[JSON] = []
        for index, (text, values) in enumerate(zip(texts, embeddings, strict=True)):
            metadata = dict(metadatas[index]) if metadatas is not None else {}
            metadata.setdefault("text", text)
            vectors.append({"id": ids[index], "values": list(values), "metadata": metadata})
        return vectors

    @staticmethod
    def _search_body(query: str, k: int, kwargs: Mapping[str, Any]) -> JSON:
        search_text = kwargs.get("search_text")
        if search_text is not None and search_text != query:
            raise ValueError(
                "Use the LangChain query argument for search text, "
                "not a separate search_text value."
            )
        body: JSON = {"query_text": query, "top_k": k, "include_documents": True}
        filter_value = kwargs.get("filter")
        filters_value = kwargs.get("filters")
        if filter_value is not None and filters_value is not None:
            raise ValueError("Use only one of filter or filters.")
        filters = filter_value if filter_value is not None else filters_value
        if filters is not None:
            body["filters"] = dict(filters)
        passthrough = {
            "advanced_filters",
            "embedding_provider",
            "embedding_model",
            "nprobe_override",
            "rerank_depth_override",
            "hybrid",
            "sparse_query",
            "alpha",
            "include_embeddings",
            "include_documents",
            "include_metadata",
            "rerank",
        }
        for key in passthrough:
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        return body
