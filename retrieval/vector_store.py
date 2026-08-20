from typing import Any
from opensearchpy import OpenSearch, helpers

from config.settings import (
    EMBEDDING_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_ENGINE,
    HNSW_M,
    HYBRID_SEARCH_ENABLED,
    OPENSEARCH_HOST,
    OPENSEARCH_INDEX,
    OPENSEARCH_PORT,
    RRF_K,
    TOP_K,
)
from ingestion.embedder import get_embedding_model

_client_instance: OpenSearch | None = None
_embedder_instance: Any | None = None


def get_client() -> OpenSearch:
    global _client_instance
    if _client_instance is None:
        _client_instance = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            use_ssl=False,
            verify_certs=False,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
    return _client_instance


def get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = get_embedding_model()
    return _embedder_instance


def create_index():
    c = get_client()
    if c.indices.exists(index=OPENSEARCH_INDEX):
        print(f"  Index '{OPENSEARCH_INDEX}' already exists — skipping creation.")
        return

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": HNSW_EF_SEARCH,
            }
        },
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "standard",
                },
                "source": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": HNSW_ENGINE,
                        "parameters": {
                            "ef_construction": HNSW_EF_CONSTRUCTION,
                            "m": HNSW_M,
                        },
                    },
                },
            }
        },
    }

    c.indices.create(index=OPENSEARCH_INDEX, body=mapping)
    print(f"  Index '{OPENSEARCH_INDEX}' created with HNSW ({HNSW_ENGINE}, M={HNSW_M}, ef_c={HNSW_EF_CONSTRUCTION})")


def add_chunks(chunks: list[dict], batch_size: int = 200):
    if not chunks:
        print("  No chunks to index.")
        return

    c = get_client()
    model = get_embedder()

    texts = [ch["text"] for ch in chunks]
    print(f"  Generating embeddings for {len(texts)} chunk(s)...")
    vectors = model.embed_documents(texts)

    actions = [
        {
            "_index": OPENSEARCH_INDEX,
            "_id": chunk["id"],
            "_source": {
                "chunk_id": chunk["id"],
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk["text"],
                "source": chunk["source"],
                "vector": vector,
            },
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    success, failed = helpers.bulk(c, actions, chunk_size=batch_size, refresh=True)
    failed_count = len(failed) if isinstance(failed, list) else 0
    print(f"  Bulk stored {success} chunk(s) in OpenSearch (failed: {failed_count})")


def vector_search(query: str, k: int = TOP_K) -> list[dict]:
    c = get_client()
    model = get_embedder()
    vector = model.embed_query(query)

    body = {
        "size": k,
        "query": {
            "knn": {
                "vector": {
                    "vector": vector,
                    "k": k,
                }
            }
        },
        "_source": ["text", "source", "chunk_id", "chunk_index"],
    }

    res = c.search(index=OPENSEARCH_INDEX, body=body)
    return [
        {
            "chunk_id": h["_source"].get("chunk_id", h["_id"]),
            "text": h["_source"]["text"],
            "source": h["_source"]["source"],
            "score": round(float(h["_score"]), 4),
            "method": "vector",
        }
        for h in res["hits"]["hits"]
    ]


def lexical_search(query: str, k: int = TOP_K) -> list[dict]:
    c = get_client()
    body = {
        "size": k,
        "query": {
            "match": {
                "text": {
                    "query": query,
                    "operator": "or",
                    "fuzziness": "AUTO",
                }
            }
        },
        "_source": ["text", "source", "chunk_id", "chunk_index"],
    }

    res = c.search(index=OPENSEARCH_INDEX, body=body)
    return [
        {
            "chunk_id": h["_source"].get("chunk_id", h["_id"]),
            "text": h["_source"]["text"],
            "source": h["_source"]["source"],
            "score": round(float(h["_score"]), 4),
            "method": "bm25",
        }
        for h in res["hits"]["hits"]
    ]


def hybrid_search(query: str, k: int = TOP_K, rrf_k: int = RRF_K) -> list[dict]:
    """
    Executes Hybrid Search combining Dense Vector Search and Lexical BM25 Search
    using Reciprocal Rank Fusion (RRF).
    """
    dense_results = vector_search(query, k=k)
    bm25_results = lexical_search(query, k=k)

    # Compute Reciprocal Rank Fusion (RRF) scores
    rrf_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    # Score dense vector ranks
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["chunk_id"]
        doc_lookup[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    # Score BM25 lexical ranks
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["chunk_id"]
        if doc_id not in doc_lookup:
            doc_lookup[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    # Sort combined results by RRF score
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    hybrid_hits = []
    for doc_id in sorted_doc_ids[:k]:
        item = dict(doc_lookup[doc_id])
        item["rrf_score"] = round(rrf_scores[doc_id], 6)
        item["score"] = item["rrf_score"]  # primary ranking score
        item["method"] = "hybrid_rrf"
        hybrid_hits.append(item)

    return hybrid_hits


def search(query: str, k: int = TOP_K) -> list[dict]:
    """
    Primary search entrypoint. Dispatches to hybrid search if enabled,
    otherwise falls back to pure vector search.
    """
    if HYBRID_SEARCH_ENABLED:
        return hybrid_search(query, k=k)
    return vector_search(query, k=k)


def delete_index():
    c = get_client()
    if c.indices.exists(index=OPENSEARCH_INDEX):
        c.indices.delete(index=OPENSEARCH_INDEX)
        print(f"  Index '{OPENSEARCH_INDEX}' deleted.")

