"""LangChain VectorStore implementation for VectorAmp."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, TypeVar, cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .client import VectorAmpHTTPClient

VST = TypeVar("VST", bound="VectorAmpVectorStore")


class VectorAmpVectorStore(VectorStore):
    """VectorAmp-backed LangChain VectorStore.

    VectorAmp hosts the embedding model for dataset text operations, so this
    class intentionally does not require a LangChain ``Embeddings`` object.
    The optional ``embedding`` argument is accepted for compatibility with
    LangChain constructors but is not used for add/search operations.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.vectoramp.com",
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        client: Optional[Any] = None,
        embedding: Optional[Embeddings] = None,
        timeout: float = 30.0,
    ) -> None:
        if dataset_id and dataset_name:
            raise ValueError("Provide only one of dataset_id or dataset_name.")
        self._external_client = client
        self._client = client or VectorAmpHTTPClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.embedding = embedding
        self.dataset_name = dataset_name
        self.dataset_id = self._resolve_dataset_id(dataset_id=dataset_id, dataset_name=dataset_name)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Optional LangChain embedding object accepted only for compatibility."""
        return self.embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed texts with VectorAmp's hosted model and add them to the dataset."""
        text_list = list(texts)
        if not text_list:
            return []
        add_ids = kwargs.pop("ids", ids)
        if kwargs:
            # Preserve forward compatibility with LangChain while not silently
            # dropping unsupported write options.
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported add_texts kwargs: {unsupported}")
        return self._add_texts(text_list, ids=add_ids, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Return documents most similar to ``query``."""
        return [
            document for document, _score in self.similarity_search_with_score(query, k=k, **kwargs)
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Return documents and VectorAmp scores for the text query."""
        response = self._search(query, k=k, **kwargs)
        return self._documents_from_search_response(response)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Async variant of :meth:`add_texts`."""
        text_list = list(texts)
        if not text_list:
            return []
        add_ids = kwargs.pop("ids", ids)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported aadd_texts kwargs: {unsupported}")
        if hasattr(self._client, "aadd_texts"):
            return await self._client.aadd_texts(
                self.dataset_id, text_list, ids=add_ids, metadatas=metadatas
            )
        return await asyncio.to_thread(self._add_texts, text_list, ids=add_ids, metadatas=metadatas)

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Async variant of :meth:`similarity_search`."""
        results = await self.asimilarity_search_with_score(query, k=k, **kwargs)
        return [document for document, _score in results]

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Async variant of :meth:`similarity_search_with_score`."""
        if hasattr(self._client, "asearch"):
            response = await self._client.asearch(self.dataset_id, query, k=k, **kwargs)
        else:
            response = await asyncio.to_thread(self._search, query, k=k, **kwargs)
        return self._documents_from_search_response(response)

    @classmethod
    def from_texts(
        cls: type[VST],
        texts: list[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> VST:
        """Create a VectorAmpVectorStore and add texts using hosted embeddings."""
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls: type[VST],
        documents: list[Document],
        embedding: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> VST:
        """Create a VectorAmpVectorStore from LangChain documents."""
        texts = [document.page_content for document in documents]
        metadatas = [dict(document.metadata) for document in documents]
        if "ids" not in kwargs:
            ids = [document.id for document in documents]
            if any(ids):
                kwargs["ids"] = ids
        return cls.from_texts(texts, embedding=embedding, metadatas=metadatas, **kwargs)

    async def afrom_texts_instance(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        *,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Convenience helper for async population of an existing store."""
        return await self.aadd_texts(texts, metadatas=metadatas, ids=ids)

    def close(self) -> None:
        """Close owned HTTP resources when using the default client."""
        if self._external_client is None and hasattr(self._client, "close"):
            self._client.close()

    async def aclose(self) -> None:
        """Async close owned HTTP resources when using the default client."""
        if self._external_client is None and hasattr(self._client, "aclose"):
            await self._client.aclose()

    def _resolve_dataset_id(self, *, dataset_id: Optional[str], dataset_name: Optional[str]) -> str:
        if dataset_id:
            return dataset_id
        if hasattr(self._client, "get_dataset_id"):
            return self._client.get_dataset_id(dataset_id=dataset_id, dataset_name=dataset_name)
        if dataset_name and hasattr(self._client, "datasets"):
            datasets = self._client.datasets.list(limit=100, offset=0).get("datasets", [])
            for dataset in datasets:
                if _mapping_get(dataset, "name") == dataset_name:
                    value = _mapping_get(dataset, "id") or _mapping_get(dataset, "dataset_id")
                    if value is not None:
                        return str(value)
            raise ValueError(f"Dataset named {dataset_name!r} was not found.")
        raise ValueError("dataset_id or dataset_name is required.")

    def _add_texts(
        self,
        texts: Sequence[str],
        *,
        ids: Optional[Sequence[str]],
        metadatas: Optional[Sequence[Mapping[str, Any]]],
    ) -> list[str]:
        if hasattr(self._client, "add_texts"):
            return self._client.add_texts(self.dataset_id, texts, ids=ids, metadatas=metadatas)
        client = cast(Any, self._client)
        result = client.datasets.add_texts(self.dataset_id, texts, ids=ids, metadatas=metadatas)
        if ids is not None:
            return [str(value) for value in ids]
        inserted_ids = result.get("ids") or result.get("vector_ids") or result.get("inserted_ids")
        if isinstance(inserted_ids, list):
            return [str(value) for value in inserted_ids]
        raise ValueError("Client add_texts response did not include inserted ids.")

    def _search(self, query: str, *, k: int, **kwargs: Any) -> Mapping[str, Any]:
        if hasattr(self._client, "search"):
            return self._client.search(self.dataset_id, query, k=k, **kwargs)
        filters = _pop_filter(kwargs)
        client = cast(Any, self._client)
        response = client.datasets.search(
            self.dataset_id,
            query,
            top_k=k,
            filters=filters,
            include_documents=kwargs.pop("include_documents", True),
            **kwargs,
        )
        return cast(Mapping[str, Any], response)

    @staticmethod
    def _documents_from_search_response(
        response: Mapping[str, Any],
    ) -> list[tuple[Document, float]]:
        raw_results = _extract_results(response)
        return [_document_from_result(result) for result in raw_results]


def _pop_filter(kwargs: dict[str, Any]) -> Optional[Mapping[str, Any]]:
    filter_value = kwargs.pop("filter", None)
    filters_value = kwargs.pop("filters", None)
    if filter_value is not None and filters_value is not None:
        raise ValueError("Use only one of filter or filters.")
    selected = filter_value if filter_value is not None else filters_value
    return cast(Optional[Mapping[str, Any]], selected)


def _extract_results(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("results", "vectors", "matches", "data"):
        value = response.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    if isinstance(response.get("result"), Mapping):
        return _extract_results(response["result"])
    return []


def _document_from_result(result: Mapping[str, Any]) -> tuple[Document, float]:
    metadata = _metadata_from_result(result)
    page_content = _content_from_result(result, metadata)
    document_id = _first_string(result, ("id", "vector_id", "document_id", "chunk_id"))
    score = _score_from_result(result)
    return Document(page_content=page_content, metadata=metadata, id=document_id), score


def _metadata_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    raw_metadata = result.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    for key in ("id", "vector_id", "document_id", "chunk_id"):
        if key in result and key not in metadata:
            metadata[key] = result[key]
    additional = result.get("additional")
    if isinstance(additional, Mapping) and "additional" not in metadata:
        metadata["additional"] = dict(additional)
    return metadata


def _content_from_result(result: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    for key in ("doc_value", "document", "text", "content", "page_content"):
        value = result.get(key)
        if isinstance(value, str):
            return value
    for key in ("doc_value", "text", "content", "page_content"):
        value = metadata.get(key)
        if isinstance(value, str):
            return value
    return ""


def _score_from_result(result: Mapping[str, Any]) -> float:
    for key in ("score", "similarity", "distance"):
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _first_string(mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return str(value)
    return None


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)
