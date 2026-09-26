"""
Ollama bge-m3 Indexing - Example Usage

This script demonstrates how to build a vector index with embeddings served
by a local Ollama server (model: bge-m3), persist it to disk, reload it, and
run semantic search queries against the rebuilt index.

Prerequisites:
    - `ollama serve` running locally (default http://localhost:11434)
    - `ollama pull bge-m3`

Output index directory: test_data/runtime/ollama_bge_m3_index/
"""

import shutil
from pathlib import Path

from semantica.vector_store import VectorStore

INDEX_DIR = Path("test_data/runtime/ollama_bge_m3_index")

# Multilingual sample documents (bge-m3 is a multilingual model)
DOCUMENTS = [
    "Semantica is a knowledge graph framework for semantic data processing.",
    "知识图谱是一种用图结构表示实体及其关系的数据组织方式。",
    "Ollama serves open-source models locally over an HTTP API.",
    "向量数据库通过嵌入向量之间的距离来衡量语义相似度。",
    "The bge-m3 embedding model supports multilingual and long-text retrieval.",
    "FastEmbed runs quantized ONNX models locally without a server.",
    "Entity resolution links records that refer to the same real-world object.",
    "检索增强生成(RAG)将检索到的文档注入到大模型的上下文中。",
]

METADATA = [
    {"source": "docs", "lang": "en", "topic": "knowledge-graph"},
    {"source": "docs", "lang": "zh", "topic": "knowledge-graph"},
    {"source": "docs", "lang": "en", "topic": "ollama"},
    {"source": "docs", "lang": "zh", "topic": "vector-search"},
    {"source": "docs", "lang": "en", "topic": "embeddings"},
    {"source": "docs", "lang": "en", "topic": "embeddings"},
    {"source": "docs", "lang": "en", "topic": "entity-resolution"},
    {"source": "docs", "lang": "zh", "topic": "rag"},
]

QUERIES = [
    "知识图谱是什么",  # expect the zh + en knowledge-graph docs
    "local model serving",  # expect the ollama doc
]


def print_results(title, results):
    print(f"\n{title}")
    for r in results:
        score = r.get("score", 0.0)
        meta = r.get("metadata", {})
        print(f"  [{score:.4f}] {meta.get('lang', '?')}: {meta.get('text', r.get('id'))}")


def build_store():
    store = VectorStore(backend="inmemory", config={"dimension": 1024})
    store.embedder.set_text_model("ollama", "bge-m3")
    print(f"Embedder: {store.embedder.get_text_method()} / bge-m3")
    return store


def main():
    print("=" * 70)
    print("Ollama bge-m3 Indexing - Example Usage")
    print("=" * 70)

    # Recreate the index from scratch
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        print(f"\nRemoved existing index at {INDEX_DIR}")

    store = build_store()
    metadata = [{**m, "text": doc} for m, doc in zip(METADATA, DOCUMENTS)]
    ids = store.add_documents(DOCUMENTS, metadata=metadata)
    print(f"Indexed {len(ids)} documents (dim={store.dimension})")

    for query in QUERIES:
        print_results(f"Query: {query!r}", store.search(query, limit=3))

    # Persist the index
    store.save(str(INDEX_DIR))
    print(f"\nIndex saved to {INDEX_DIR}")

    # Reload and verify the rebuilt index answers the same queries
    reloaded = build_store()
    reloaded.load(str(INDEX_DIR))
    print(f"Reloaded index: {len(reloaded.vectors)} vectors")

    for query in QUERIES:
        print_results(f"Query (reloaded): {query!r}", reloaded.search(query, limit=3))


if __name__ == "__main__":
    main()
