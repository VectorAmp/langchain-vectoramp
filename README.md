# langchain-vectoramp

LangChain `VectorStore`, retriever, and loader integration for [VectorAmp](https://vectoramp.com), backed by VectorAmp-hosted embeddings and SABLE datasets.

## Installation

```bash
pip install langchain-vectoramp
```

## Usage

```python
from langchain_vectoramp import VectorAmpVectorStore

store = VectorAmpVectorStore(
    api_key="va_...",              # or VECTORAMP_API_KEY
    dataset_id="dataset-id",      # or dataset_name="docs"
)

store.add_texts(
    ["VectorAmp is built for billion-scale vector search."],
    metadatas=[{"source": "readme"}],
)

docs = store.similarity_search(
    "What is VectorAmp built for?",
    k=3,
    filter={"source": "readme"},
    rerank={"enabled": True},  # vectoramp / VectorAmp-Rerank-v1
)
```

VectorAmp owns embedding for text add/search. A LangChain `Embeddings` object is accepted for constructor compatibility, but normal operations use VectorAmp text endpoints.

## Retriever

```python
retriever = store.as_retriever(
    search_kwargs={"k": 5, "filter": {"tenant": "acme"}}
)
results = retriever.invoke("find relevant docs")
```

## Intelligence (RAG answers)

`VectorAmpVectorStore` retrieves documents; `VectorAmpIntelligence` returns a
generated answer with sources. The simplest usage needs only a question, and
queries run across all your datasets by default:

```python
from langchain_vectoramp import VectorAmpIntelligence

intel = VectorAmpIntelligence()  # uses VECTORAMP_API_KEY, all datasets, sources on
print(intel.ask("What changed in the Q4 planning docs?"))
```

Multi-turn "just works": pass prior LangChain messages and they are forwarded as
conversation history automatically. You control how many previous messages are
included simply by how many you pass.

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

# Need sources/metadata too?
intel.ask_with_sources("What are the contract termination terms?")
```

Scope to one dataset or tune retrieval only when you need to:

```python
VectorAmpIntelligence(dataset_name="contracts", top_k=8)
```

## Loader

```python
from langchain_vectoramp import VectorAmpLoader

loader = VectorAmpLoader(
    api_key="va_...",
    dataset_id="dataset-id",
    query="contract renewal terms",  # required in v1
    filter={"source": "contracts"},
    k=10,
    metadata={"loaded_from": "vectoramp"},
)

documents = loader.load()
# or stream lazily:
for document in loader.lazy_load():
    print(document.page_content)
```

`VectorAmpLoader` now uses the dataset documents API when `query` is omitted, listing retained source documents and downloading their originals. Provide `query` to keep the semantic-search-backed behavior and return the top `k` matching documents instead.

## API

`VectorAmpVectorStore` supports:

- `add_texts` / LangChain `add_documents`
- `similarity_search`
- `similarity_search_with_score`
- async `aadd_texts`, `asimilarity_search`, `asimilarity_search_with_score`
- `from_texts` and `from_documents`
- metadata filters via `filter={...}` or `filters={...}` plus VectorAmp-native search kwargs such as `advanced_filters`, `include_metadata`, `hybrid`, `sparse_query`, and `alpha`

`VectorAmpLoader` supports:

- constructor args `api_key`, `base_url`, `dataset_id`, `client`, optional `query`, `filter`, `k`, and `metadata`
- `load()` and idiomatic `lazy_load()`
- the same search kwargs passthrough used by the vector store

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
mypy
```

Tests use mocked clients/transports only; no live VectorAmp API calls are made.
