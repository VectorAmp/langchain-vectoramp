"""LangChain document loader for VectorAmp dataset documents or search results."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .client import VectorAmpHTTPClient
from .vectorstores import VectorAmpVectorStore


class VectorAmpLoader(BaseLoader):
    """Load LangChain documents from VectorAmp retained documents or search results.

    When ``query`` is omitted, the loader uses VectorAmp's public
    ``/datasets/{id}/documents`` endpoint to iterate retained source documents.
    When ``query`` is provided, it remains semantic-search-backed and returns the
    top ``k`` matching documents.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.vectoramp.com",
        dataset_id: str,
        client: Optional[Any] = None,
        query: Optional[str] = None,
        filter: Optional[Mapping[str, Any]] = None,
        k: int = 10,
        metadata: Optional[Mapping[str, Any]] = None,
        timeout: float = 30.0,
        **search_kwargs: Any,
    ) -> None:
        self.dataset_id = dataset_id
        self.query = query
        self.filter = dict(filter) if filter is not None else None
        self.k = k
        self.metadata = dict(metadata) if metadata is not None else {}
        self.search_kwargs = dict(search_kwargs)
        self._external_client = client
        self._client = client or VectorAmpHTTPClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def lazy_load(self) -> Iterator[Document]:
        """Yield retained dataset documents or semantic-search matches."""
        if not self.query:
            yield from self._lazy_load_documents()
            return

        kwargs = dict(self.search_kwargs)
        if self.filter is not None:
            kwargs["filter"] = self.filter
        kwargs.setdefault("include_documents", True)
        kwargs.setdefault("include_metadata", True)
        store = VectorAmpVectorStore(client=self._client, dataset_id=self.dataset_id)
        for document in store.similarity_search(self.query, k=self.k, **kwargs):
            if self.metadata:
                document.metadata = {**self.metadata, **document.metadata}
            yield document

    def _lazy_load_documents(self) -> Iterator[Document]:
        if not hasattr(self._client, "list_documents"):
            raise ValueError(
                "VectorAmpLoader requires query when the client cannot list documents."
            )

        cursor: Optional[str] = None
        while True:
            page = self._client.list_documents(
                self.dataset_id, limit=self.k, cursor=cursor, status="ready"
            )
            documents = page.get("documents") or page.get("data") or page.get("items") or []
            if not documents:
                break
            for item in documents:
                if not isinstance(item, Mapping):
                    continue
                content = self._document_content(item)
                if content is None:
                    if not hasattr(self._client, "download_document"):
                        continue
                    document_id = item.get("id") or item.get("document_id")
                    if document_id is None:
                        continue
                    content = self._client.download_document(
                        self.dataset_id, str(document_id)
                    ).decode("utf-8", errors="replace")
                metadata = {
                    key: value
                    for key, value in item.items()
                    if key not in {"text", "content", "page_content", "doc_value"}
                }
                if self.metadata:
                    metadata = {**self.metadata, **metadata}
                doc_id = item.get("id") or item.get("document_id")
                yield Document(
                    page_content=str(content),
                    metadata=metadata,
                    id=str(doc_id) if doc_id is not None else None,
                )

            cursor = page.get("next_cursor") or page.get("nextCursor")
            if not cursor:
                break

    @staticmethod
    def _document_content(item: Mapping[str, Any]) -> Any:
        for key in ("text", "content", "page_content", "doc_value"):
            value = item.get(key)
            if value is not None:
                return value
        return None

    def load(self) -> list[Document]:
        """Return all documents from :meth:`lazy_load`."""
        return list(self.lazy_load())

    def close(self) -> None:
        """Close owned HTTP resources."""
        if self._external_client is None and hasattr(self._client, "close"):
            self._client.close()
