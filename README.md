# langchain-vectoramp

LangChain `VectorStore`, retriever, and document loader integration for
[VectorAmp](https://vectoramp.com), backed by VectorAmp-hosted embeddings and
SABLE datasets.

VectorAmp embeds your text **server-side**, so you never have to wire up a
LangChain `Embeddings` object: text add and search call VectorAmp's hosted
model and SABLE index directly.

## Installation

```bash
pip install langchain-vectoramp
```

Set your API key once in the environment (every class reads it automatically):

```bash
export VECTORAMP_API_KEY="vsk_..."
```

## Quickstart

```python
from langchain_vectoramp import VectorAmpVectorStore

# API key comes from VECTORAMP_API_KEY; base_url defaults to production.
store = VectorAmpVectorStore(dataset_name="docs")   # or dataset_id="..."

store.add_texts(
    ["VectorAmp is built for billion-scale vector search."],
    metadatas=[{"source": "readme"}],
)

docs = store.similarity_search(
    "What is VectorAmp built for?",
    k=3,
    filter={"source": "readme"},
    rerank=True,                # expands to the VectorAmp-Rerank-v1 rerank object
)
```

`add_texts` embeds each text with the dataset's hosted model, copies the source
text into `metadata.text`, generates ids when you omit them, and inserts the
vectors in one call.

### Vector ids: strings or integers

`ids` accepts strings **or** integers. Integer ids are preserved and sent as
JSON numbers, so the server stores exactly what you passed:

```python
store.add_texts(["a", "b"], ids=[1, 2])      # -> [1, 2], stored as numbers
store.add_texts(["c"], ids=["doc-c"])        # -> ["doc-c"]
```

## Retriever

`as_retriever()` is inherited from the LangChain `VectorStore` base class and
works out of the box:

```python
retriever = store.as_retriever(
    search_kwargs={"k": 5, "filter": {"tenant": "acme"}},
)
results = retriever.invoke("find relevant docs")
```

## Intelligence (RAG answers)

`VectorAmpVectorStore` retrieves documents; `VectorAmpIntelligence` returns a
generated answer with sources. The simplest usage needs only a question, and
queries run across **all** your datasets by default:

```python
from langchain_vectoramp import VectorAmpIntelligence

intel = VectorAmpIntelligence()   # uses VECTORAMP_API_KEY, all datasets, sources on
print(intel.ask("What changed in the Q4 planning docs?"))
```

Multi-turn "just works": pass prior LangChain messages and they are forwarded as
conversation history automatically. You decide how many previous turns to include
simply by how many you pass.

```python
from langchain_core.messages import HumanMessage, AIMessage

history = [
    HumanMessage("What is VectorAmp?"),
    AIMessage("A vector database platform."),
]
intel.ask("Does it support hybrid search?", history=history)
```

It is also a `Runnable`, so it composes in LCEL chains and works with
`RunnableWithMessageHistory` — pass a list of messages and the last one is taken
as the question:

```python
intel.invoke("Summarize the docs")
intel.invoke([HumanMessage("hi"), AIMessage("hello"), HumanMessage("and reranking?")])

# Need the sources/metadata too?
intel.ask_with_sources("What are the contract termination terms?")
```

Scope to one dataset or tune retrieval only when you need to:

```python
intel = VectorAmpIntelligence(dataset_name="contracts", top_k=8)
```

## Loader

```python
from langchain_vectoramp import VectorAmpLoader

loader = VectorAmpLoader(
    dataset_id="dataset-id",
    metadata={"loaded_from": "vectoramp"},
)

documents = loader.load()
# or stream lazily:
for document in loader.lazy_load():
    print(document.page_content)
```

When `query` is omitted the loader lists the dataset's retained source documents
via `GET /datasets/{id}/documents` (downloading originals as needed). Provide a
`query` to switch to semantic search and return the top `k` matching documents:

```python
loader = VectorAmpLoader(
    dataset_id="dataset-id",
    query="contract renewal terms",
    filter={"source": "contracts"},
    k=10,
)
```

## A note on embeddings

VectorAmp embeds server-side. Every class accepts an optional LangChain
`Embeddings` object (`embedding=...` on the vector store) **for constructor
compatibility only** — it is intentionally ignored for add and search, which
always use VectorAmp's hosted model. You never need to provide one.

## Method reference

### `VectorAmpVectorStore`

Constructor:

```python
VectorAmpVectorStore(
    *,
    api_key: str | None = None,        # default: env VECTORAMP_API_KEY
    base_url: str = "https://api.vectoramp.com",
    dataset_id: str | None = None,     # provide exactly one of dataset_id / dataset_name
    dataset_name: str | None = None,   # resolved to an id via GET /datasets
    client: VectorAmpHTTPClient | None = None,
    embedding: Embeddings | None = None,   # accepted but ignored (server-side embedding)
    timeout: float = 30.0,
)
```

| Method | Required | Optional (defaults) | Returns |
|---|---|---|---|
| `add_texts(texts, metadatas=None, *, ids=None, **kwargs)` | `texts` | `metadatas`, `ids` (auto-generated UUIDs; str or int) | `list[str \| int]` ids |
| `aadd_texts(texts, metadatas=None, *, ids=None, **kwargs)` | `texts` | `metadatas`, `ids` | `list[str \| int]` |
| `similarity_search(query, k=4, **kwargs)` | `query` | `k` (4), `filter`/`filters`, `hybrid`, `sparse_query`, `alpha`, `rerank`, `advanced_filters`, `include_metadata`, `embedding_provider`, `embedding_model`, `nprobe_override`, `rerank_depth_override` | `list[Document]` |
| `similarity_search_with_score(query, k=4, **kwargs)` | `query` | same as above | `list[tuple[Document, float]]` |
| `asimilarity_search(query, k=4, **kwargs)` | `query` | same as above | `list[Document]` |
| `asimilarity_search_with_score(query, k=4, **kwargs)` | `query` | same as above | `list[tuple[Document, float]]` |
| `as_retriever(**kwargs)` | — | `search_kwargs` (e.g. `{"k": 5, "filter": {...}}`), `search_type` | `VectorStoreRetriever` (inherited) |
| `from_texts(texts, embedding=None, metadatas=None, *, ids=None, **kwargs)` *(classmethod)* | `texts`, plus `dataset_id` or `dataset_name` in `kwargs` | `embedding` (ignored), `metadatas`, `ids` | `VectorAmpVectorStore` |
| `from_documents(documents, embedding=None, **kwargs)` *(classmethod)* | `documents`, plus `dataset_id` or `dataset_name` | `embedding` (ignored) | `VectorAmpVectorStore` |
| `afrom_texts_instance(texts, metadatas=None, *, ids=None)` | `texts` | `metadatas`, `ids` | `list[str \| int]` |
| `close()` / `aclose()` | — | — | `None` |

Properties: `dataset_id` (resolved id), `dataset_name` (original name, if given),
`embeddings` (the unused `embedding` arg).

Filtering: pass `filter={...}` (the LangChain convention, preferred) or
`filters={...}` (accepted for SDK parity) — use only one.

### `VectorAmpIntelligence`

Constructor:

```python
VectorAmpIntelligence(
    *,
    dataset_id: str | None = None,     # default: query across all datasets ("all")
    dataset_name: str | None = None,
    api_key: str | None = None,        # default: env VECTORAMP_API_KEY
    base_url: str = "https://api.vectoramp.com",
    client: VectorAmpHTTPClient | None = None,
    top_k: int | None = None,          # default: server default (5)
    include_sources: bool = True,
)
```

| Method | Required | Optional (defaults) | Returns |
|---|---|---|---|
| `ask(query, *, history=None, **overrides)` | `query` | `history`, `dataset_id`, `include_sources`, `top_k` | `str` answer |
| `aask(query, *, history=None, **overrides)` | `query` | same as `ask` | `str` |
| `ask_with_sources(query, *, history=None, **overrides)` | `query` | same as `ask` | `dict` (answer, sources, metadata) |
| `invoke(input, config=None, **kwargs)` | `input` | — | `str` |
| `ainvoke(input, config=None, **kwargs)` | `input` | — | `str` |

`invoke`/`ainvoke` accept a plain string, a dict (`{query|question|input,
history|chat_history|messages}`), or a list of LangChain messages (the last is
the question, the rest become conversation history). LangChain message types map
to API roles: `human → user`, `ai → assistant`, `system → system`, `tool → tool`.

### `VectorAmpLoader`

Constructor:

```python
VectorAmpLoader(
    *,
    dataset_id: str,                   # required
    api_key: str | None = None,        # default: env VECTORAMP_API_KEY
    base_url: str = "https://api.vectoramp.com",
    client: VectorAmpHTTPClient | None = None,
    query: str | None = None,          # omit -> list documents; provide -> semantic search
    filter: Mapping | None = None,
    k: int = 10,
    metadata: Mapping | None = None,   # merged into every loaded Document's metadata
    timeout: float = 30.0,
    **search_kwargs,                   # forwarded to search when query is set
)
```

| Method | Returns | Behavior |
|---|---|---|
| `load()` | `list[Document]` | Materializes all documents |
| `lazy_load()` | `Iterator[Document]` | Streams documents (paginates the documents API or runs one search) |
| `close()` | `None` | Closes owned HTTP resources |

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
mypy
```

Tests use mocked clients/transports only; no live VectorAmp API calls are made.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
