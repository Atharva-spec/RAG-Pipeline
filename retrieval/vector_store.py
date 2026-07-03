from opensearchpy import OpenSearch

from ingestion.embedder import get_embedding_model
from config.settings import (
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_INDEX,
    EMBEDDING_DIM,
    TOP_K,
)


def _client():
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        use_ssl=False,
    )


def create_index():
    c = _client()
    if c.indices.exists(index=OPENSEARCH_INDEX):
        print(" Index already exists — skipping")
        return

    # Exact KNN — no method block needed
    c.indices.create(
        index=OPENSEARCH_INDEX,
        body={
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "vector": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIM,
                        # no method block = exact KNN search
                    },
                }
            },
        },
    )
    print(f" Index '{OPENSEARCH_INDEX}' created with exact KNN")


def add_chunks(chunks: list[dict]):
    c = _client()
    model = get_embedding_model()
    texts = [ch["text"] for ch in chunks]
    vectors = model.embed_documents(texts)

    for chunk, vector in zip(chunks, vectors):
        c.index(
            index=OPENSEARCH_INDEX,
            id=chunk["id"],
            body={
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "vector": vector,
            },
            refresh=True,
        )

    print(f" {len(chunks)} chunks stored in OpenSearch")


def search(query: str, k: int = TOP_K) -> list[dict]:
    c = _client()
    model = get_embedding_model()
    vector = model.embed_query(query)

    res = c.search(
        index=OPENSEARCH_INDEX,
        body={
            "size": k,
            "query": {
                "knn": {"vector": {"vector": vector, "k": k}}
            },
            "_source": ["text", "source"],
        },
    )

    return [
        {
            "text": h["_source"]["text"],
            "source": h["_source"]["source"],
            "score": round(h["_score"], 4),
        }
        for h in res["hits"]["hits"]
    ]


def delete_index():
    c = _client()
    if c.indices.exists(index=OPENSEARCH_INDEX):
        c.indices.delete(index=OPENSEARCH_INDEX)
        print(" Index deleted")
