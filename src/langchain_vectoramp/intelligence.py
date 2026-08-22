"""LangChain Intelligence (RAG) interface for VectorAmp.

Designed for minimal-friction usage: a one-liner answers a question, queries run
across all datasets by default, and multi-turn conversations "just work" because
LangChain message history is mapped to the API's ``conversation_history``
automatically. The caller decides how many prior messages to include simply by
how many they pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

from .client import VectorAmpHTTPClient

JSON = dict[str, Any]
Turn = dict[str, str]
Message = Union[BaseMessage, dict[str, Any], str]

# LangChain message ``type`` -> Intelligence ``role``.
_ROLE_BY_TYPE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # LangChain content blocks
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "".join(parts)
    return str(content)


def _to_turn(message: Message) -> Turn:
    if isinstance(message, BaseMessage):
        return {
            "role": _ROLE_BY_TYPE.get(message.type, "user"),
            "content": _content_to_text(message.content),
        }
    if isinstance(message, dict):
        role = message.get("role") or _ROLE_BY_TYPE.get(str(message.get("type", "")), "user")
        return {"role": role, "content": _content_to_text(message.get("content", ""))}
    return {"role": "user", "content": str(message)}


def _as_list(value: Union[None, str, Sequence[str]]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_dataset_ids(dataset_ids: Union[None, str, Sequence[str]]) -> list[str]:
    """Drop blanks and the retired ``"all"`` sentinel from a dataset scope.

    ``"all"`` used to be the way to ask across everything; that is now spelled by
    omitting ``dataset_ids`` entirely, and the API filters the sentinel out anyway.
    """
    return [
        str(d).strip()
        for d in _as_list(dataset_ids)
        if str(d).strip() and str(d).strip() != "all"
    ]


class VectorAmpIntelligence(Runnable[Any, str]):
    """Ask VectorAmp Intelligence (RAG) from LangChain.

    The simplest usage needs only a question::

        intel = VectorAmpIntelligence()          # uses VECTORAMP_API_KEY, all datasets
        print(intel.ask("What changed in Q4 planning?"))

    Scope a question to one or more datasets with ``dataset_ids`` (or
    ``dataset_names``); leave them unset to search everything the key can see::

        intel = VectorAmpIntelligence(dataset_ids=["ds_contracts", "ds_invoices"])
        print(intel.ask("Which contracts renew in Q4?"))

    Multi-turn follow-ups: pass prior LangChain messages (or plain dicts); they
    are forwarded as ``conversation_history`` automatically::

        from langchain_core.messages import HumanMessage, AIMessage
        history = [HumanMessage("What is VectorAmp?"), AIMessage("A vector DB platform.")]
        intel.ask("Does it support hybrid search?", history=history)

    As a LangChain ``Runnable`` it composes in chains and works with
    ``RunnableWithMessageHistory`` (pass a list of messages; the last human
    message is the question and the rest become conversation history)::

        intel.invoke("Summarize the docs")
        intel.invoke([HumanMessage("hi"), AIMessage("hello"), HumanMessage("and bye?")])
    """

    def __init__(
        self,
        *,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_ids: Optional[Sequence[str]] = None,
        dataset_names: Optional[Sequence[str]] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.vectoramp.com",
        client: Optional[VectorAmpHTTPClient] = None,
        top_k: Optional[int] = None,
        include_sources: bool = True,
    ) -> None:
        self._client = client or VectorAmpHTTPClient(api_key=api_key, base_url=base_url)
        # Singular and plural are both accepted and merged: the singular forms are
        # the natural spelling for the common one-dataset case, and both end up in
        # the same dataset_ids list on the wire.
        self._dataset_ids = [*_as_list(dataset_id), *_as_list(dataset_ids)]
        self._dataset_names = [*_as_list(dataset_name), *_as_list(dataset_names)]
        self._top_k = top_k
        self._include_sources = include_sources

    def _datasets(self) -> list[str]:
        """The configured scope, as dataset ids.

        An empty list means "every dataset the caller can see", which is what an
        absent ``dataset_ids`` says to the API. Names are resolved to ids only
        when given, so the default costs no extra request.
        """
        scope = list(self._dataset_ids)
        for name in self._dataset_names:
            scope.append(self._client.get_dataset_id(dataset_id=None, dataset_name=name))
        return _normalize_dataset_ids(scope)

    def _body(
        self,
        query: str,
        history: Optional[Sequence[Message]],
        overrides: dict[str, Any],
    ) -> JSON:
        if "dataset_id" in overrides:
            raise TypeError(
                "dataset_id is retired. Pass dataset_ids=[...], or omit it to search "
                "every dataset you can see."
            )
        body: JSON = {
            "query": query,
            "include_sources": overrides.get("include_sources", self._include_sources),
            "stream": False,
        }
        scope = (
            _normalize_dataset_ids(overrides["dataset_ids"])
            if "dataset_ids" in overrides
            else self._datasets()
        )
        # Omitted, not empty: an absent field is how the API says "every dataset".
        if scope:
            body["dataset_ids"] = scope
        top_k = overrides.get("top_k", self._top_k)
        if top_k is not None:
            body["top_k"] = top_k
        turns = [_to_turn(message) for message in (history or [])]
        if turns:
            body["conversation_history"] = turns
        return body

    def ask(
        self,
        query: str,
        *,
        history: Optional[Sequence[Message]] = None,
        **overrides: Any,
    ) -> str:
        """Ask a question and return the answer text."""
        response = self._client.intelligence_query(self._body(query, history, overrides))
        return str(response.get("answer", ""))

    async def aask(
        self,
        query: str,
        *,
        history: Optional[Sequence[Message]] = None,
        **overrides: Any,
    ) -> str:
        """Async variant of :meth:`ask`."""
        data = await self._client.aintelligence_query(self._body(query, history, overrides))
        return str(data.get("answer", ""))

    def ask_with_sources(
        self,
        query: str,
        *,
        history: Optional[Sequence[Message]] = None,
        **overrides: Any,
    ) -> JSON:
        """Return the full Intelligence response (answer, sources, metadata)."""
        return self._client.intelligence_query(self._body(query, history, overrides))

    # --- Runnable interface -------------------------------------------------
    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        query, history = self._split(input)
        return self.ask(query, history=history)

    async def ainvoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        query, history = self._split(input)
        return await self.aask(query, history=history)

    @staticmethod
    def _split(input: Any) -> tuple[str, list[Message]]:
        """Normalize Runnable input into (query, history).

        Accepts a plain string, a dict
        ({query|question|input, history|chat_history|messages}), or a sequence of
        messages where the last one is the question.
        """
        if isinstance(input, str):
            return input, []
        if isinstance(input, dict):
            query = input.get("query") or input.get("question") or input.get("input") or ""
            history = list(
                input.get("history") or input.get("chat_history") or input.get("messages") or []
            )
            if not query and history:
                last = history.pop()
                return _to_turn(last)["content"], history
            return str(query), history
        if isinstance(input, Sequence):
            messages = list(input)
            if not messages:
                return "", []
            last = messages[-1]
            return _to_turn(last)["content"], messages[:-1]
        return str(input), []
