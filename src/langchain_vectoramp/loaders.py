"""LangChain document loader for VectorAmp search results."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .client import VectorAmpHTTPClient
from .vectorstores import VectorAmpVectorStore


class VectorAmpLoader(BaseLoader):
    """Load LangChain documents from VectorAmp semantic search results.

    VectorAmp does not currently expose a full public dataset export/list-documents
    endpoint. This v1 loader is intentionally search-backed: provide ``query`` to
    load the top ``k`` matching documents from a dataset.
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
        """Yield documents returned by VectorAmp semantic search."""
        if not self.query:
            raise ValueError(
                "VectorAmpLoader v1 requires query because VectorAmp does not "
                "currently expose a public full-dataset export endpoint."
            )
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

    def load(self) -> list[Document]:
        """Return all documents from :meth:`lazy_load`."""
        return list(self.lazy_load())

    def close(self) -> None:
        """Close owned HTTP resources."""
        if self._external_client is None and hasattr(self._client, "close"):
            self._client.close()
