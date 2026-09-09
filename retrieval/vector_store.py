import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from config.settings import (
    DATABASE_URL,
    EMBEDDING_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    HYBRID_SEARCH_ENABLED,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_TABLE,
    POSTGRES_USER,
    RRF_K,
    TOP_K,
)
logger = logging.getLogger(__name__)

_embedder_instance: Any = None


def get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        from ingestion.embedder import get_embedding_model
        _embedder_instance = get_embedding_model()
    return _embedder_instance


def get_connection() -> psycopg.Connection:
    """
    Creates and returns a psycopg connection configured with pgvector and dict_row.
    Automatically creates the vector extension on first run if needed.
    """
    conn = psycopg.connect(
        conninfo=DATABASE_URL,
        row_factory=dict_row,
        autocommit=True,
    )
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
    return conn


def create_index():
    """
    Initializes PostgreSQL database schema:
    1. Installs 'vector' extension.
    2. Creates 'document_chunks' table with generated tsvector column.
    3. Builds HNSW cosine index, GIN full-text search index, and B-Tree source index.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 2. Create chunks table
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {POSTGRES_TABLE} (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    doc_hash TEXT,
                    content_hash TEXT,
                    chunk_index INTEGER,
                    total_chunks INTEGER,
                    text TEXT,
                    parent_id TEXT,
                    parent_text TEXT,
                    source TEXT,
                    file_type TEXT,
                    file_size_bytes BIGINT,
                    last_modified TEXT,
                    section_header TEXT,
                    char_count INTEGER,
                    word_count INTEGER,
                    strategy TEXT,
                    vector vector({EMBEDDING_DIM}),
                    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED
                );
                """
            )

            # 3. Create HNSW Cosine Vector index
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{POSTGRES_TABLE}_hnsw
                ON {POSTGRES_TABLE} USING hnsw (vector vector_cosine_ops)
                WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
                """
            )

            # 4. Create GIN Full-Text Search index
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{POSTGRES_TABLE}_tsv
                ON {POSTGRES_TABLE} USING gin (tsv);
                """
            )

            # 5. Create Source lookup index for fast incremental pruning
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{POSTGRES_TABLE}_source
                ON {POSTGRES_TABLE} (source);
                """
            )
            print(f"  PostgreSQL table '{POSTGRES_TABLE}' initialized with HNSW ({EMBEDDING_DIM}d, M={HNSW_M}) & GIN full-text indexes.")


def add_chunks(chunks: list[dict], batch_size: int = 200):
    if not chunks:
        print("  No chunks to index.")
        return

    model = get_embedder()
    texts = [ch["text"] for ch in chunks]
    print(f"  Generating embeddings for {len(texts)} chunk(s)...")
    vectors = model.embed_documents(texts)

    insert_sql = f"""
        INSERT INTO {POSTGRES_TABLE} (
            id, doc_id, doc_hash, content_hash, chunk_index, total_chunks,
            text, parent_id, parent_text, source, file_type, file_size_bytes,
            last_modified, section_header, char_count, word_count, strategy, vector
        ) VALUES (
            %(id)s, %(doc_id)s, %(doc_hash)s, %(content_hash)s, %(chunk_index)s, %(total_chunks)s,
            %(text)s, %(parent_id)s, %(parent_text)s, %(source)s, %(file_type)s, %(file_size_bytes)s,
            %(last_modified)s, %(section_header)s, %(char_count)s, %(word_count)s, %(strategy)s, %(vector)s
        )
        ON CONFLICT (id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id,
            doc_hash = EXCLUDED.doc_hash,
            content_hash = EXCLUDED.content_hash,
            chunk_index = EXCLUDED.chunk_index,
            total_chunks = EXCLUDED.total_chunks,
            text = EXCLUDED.text,
            parent_id = EXCLUDED.parent_id,
            parent_text = EXCLUDED.parent_text,
            source = EXCLUDED.source,
            file_type = EXCLUDED.file_type,
            file_size_bytes = EXCLUDED.file_size_bytes,
            last_modified = EXCLUDED.last_modified,
            section_header = EXCLUDED.section_header,
            char_count = EXCLUDED.char_count,
            word_count = EXCLUDED.word_count,
            strategy = EXCLUDED.strategy,
            vector = EXCLUDED.vector;
    """

    records = []
    for ch, vec in zip(chunks, vectors):
        records.append({
            "id": ch["id"],
            "doc_id": ch.get("doc_id", ch.get("source", "")),
            "doc_hash": ch.get("doc_hash", ""),
            "content_hash": ch.get("content_hash", ""),
            "chunk_index": ch.get("chunk_index", 0),
            "total_chunks": ch.get("total_chunks", len(chunks)),
            "text": ch["text"],
            "parent_id": ch.get("parent_id"),
            "parent_text": ch.get("parent_text", ch["text"]),
            "source": ch.get("source", "Unknown"),
            "file_type": ch.get("file_type", ""),
            "file_size_bytes": ch.get("file_size_bytes", 0),
            "last_modified": ch.get("last_modified", ""),
            "section_header": ch.get("section_header", "General"),
            "char_count": ch.get("char_count", len(ch["text"])),
            "word_count": ch.get("word_count", len(ch["text"].split())),
            "strategy": ch.get("strategy", "hierarchical"),
            "vector": vec,
        })

    with get_connection() as conn:
        with conn.cursor() as cur:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                cur.executemany(insert_sql, batch)

    print(f"  Stored {len(records)} chunk(s) in PostgreSQL '{POSTGRES_TABLE}'")


def _format_row(row: dict, score: float, method: str) -> dict:
    return {
        "chunk_id": row.get("id", ""),
        "doc_id": row.get("doc_id", row.get("source")),
        "doc_hash": row.get("doc_hash", ""),
        "content_hash": row.get("content_hash", ""),
        "text": row["text"],
        "parent_id": row.get("parent_id"),
        "parent_text": row.get("parent_text", row["text"]),
        "source": row.get("source", "Unknown"),
        "file_type": row.get("file_type", ""),
        "file_size_bytes": row.get("file_size_bytes", 0),
        "last_modified": row.get("last_modified", ""),
        "section_header": row.get("section_header", "General"),
        "chunk_index": row.get("chunk_index", 0),
        "total_chunks": row.get("total_chunks", 0),
        "char_count": row.get("char_count", len(row["text"])),
        "word_count": row.get("word_count", len(row["text"].split())),
        "strategy": row.get("strategy", "hierarchical"),
        "score": round(float(score), 4),
        "method": method,
    }


def vector_search(query: str, k: int = TOP_K) -> list[dict]:
    model = get_embedder()
    vector = model.embed_query(query)

    query_sql = f"""
        SELECT
            id, doc_id, doc_hash, content_hash, chunk_index, total_chunks,
            text, parent_id, parent_text, source, file_type, file_size_bytes,
            last_modified, section_header, char_count, word_count, strategy,
            1.0 - (vector <=> %(vec)s::vector) AS score
        FROM {POSTGRES_TABLE}
        ORDER BY vector <=> %(vec)s::vector
        LIMIT %(limit)s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH};")
            cur.execute(query_sql, {"vec": vector, "limit": k})
            rows = cur.fetchall()
            return [_format_row(r, r["score"], "vector") for r in rows]


def lexical_search(query: str, k: int = TOP_K) -> list[dict]:
    # Strip conversational frames so full-text search doesn't require conversational stop words
    clean_query = re.sub(
        r"^(?:can you (?:please )?(?:tell|explain|describe|give|share|provide)(?: to)?(?: me)?(?: more)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?|could you (?:please )?(?:tell|explain|describe|give|share|provide)(?: to)?(?: me)?(?: more)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?|please (?:tell|explain|describe|give|share|provide)(?: to)?(?: me)?(?: more)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?|tell me(?: more)?(?: about| regarding| on)?|talk about|explain to me(?: about)?|what can you tell me(?: about| regarding)?|what do you know about|what about|how about|i (?:just )?(?:want|would like|wish) to know(?: more)?(?: about| on| regarding)?|give me(?: some)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?|information (?:about|on|regarding)|details (?:about|on|regarding))\s+",
        "",
        query.strip(),
        flags=re.IGNORECASE,
    ).strip()
    if not clean_query:
        clean_query = query.strip()

    query_sql = f"""
        SELECT
            id, doc_id, doc_hash, content_hash, chunk_index, total_chunks,
            text, parent_id, parent_text, source, file_type, file_size_bytes,
            last_modified, section_header, char_count, word_count, strategy,
            ts_rank_cd(tsv, plainto_tsquery('english', %(query)s)) AS score
        FROM {POSTGRES_TABLE}
        WHERE tsv @@ plainto_tsquery('english', %(query)s)
        ORDER BY score DESC
        LIMIT %(limit)s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query_sql, {"query": clean_query, "limit": k})
            rows = cur.fetchall()
            if rows:
                return [_format_row(r, r["score"], "bm25") for r in rows]

            # Fallback: OR across significant keywords if plainto had 0 hits
            stopwords = {
                "tell", "about", "what", "how", "when", "where", "which", "who",
                "the", "and", "for", "with", "can", "you", "please", "give", "more",
                "details", "info", "information", "describe", "explain", "want", "know",
                "some", "regarding", "from", "into", "over", "after", "before", "is",
                "are", "was", "were", "do", "does", "did", "meaning", "meanings",
                "use", "uses", "used", "definition", "definitions", "purpose", "overview",
            }
            tokens = [w for w in re.findall(r"[a-zA-Z0-9_\-]+", clean_query) if len(w) > 1 and w.lower() not in stopwords]
            if tokens:
                or_q = " | ".join(tokens)
                fallback_sql = f"""
                    SELECT
                        id, doc_id, doc_hash, content_hash, chunk_index, total_chunks,
                        text, parent_id, parent_text, source, file_type, file_size_bytes,
                        last_modified, section_header, char_count, word_count, strategy,
                        ts_rank_cd(tsv, to_tsquery('english', %(query)s)) AS score
                    FROM {POSTGRES_TABLE}
                    WHERE tsv @@ to_tsquery('english', %(query)s)
                    ORDER BY score DESC
                    LIMIT %(limit)s;
                """
                cur.execute(fallback_sql, {"query": or_q, "limit": k})
                fallback_rows = cur.fetchall()
                if fallback_rows:
                    return [_format_row(r, r["score"], "bm25_fallback") for r in fallback_rows]
            return []


def hybrid_search(query: str, k: int = TOP_K, rrf_k: int = RRF_K) -> list[dict]:
    """
    Executes Hybrid Search combining Dense Vector Search and Lexical BM25 Search
    concurrently using ThreadPoolExecutor and Reciprocal Rank Fusion (RRF).
    """
    fetch_k = max(k * 3, 24)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dense = executor.submit(vector_search, query, fetch_k)
        future_bm25 = executor.submit(lexical_search, query, fetch_k)
        dense_results = future_dense.result()
        bm25_results = future_bm25.result()

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
        item["score"] = item["rrf_score"]
        item["method"] = "hybrid_rrf"
        hybrid_hits.append(item)

    return hybrid_hits


def search(query: str, k: int = TOP_K) -> list[dict]:
    if HYBRID_SEARCH_ENABLED:
        return hybrid_search(query, k=k)
    return vector_search(query, k=k)


def delete_index():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {POSTGRES_TABLE} CASCADE;")
            print(f"  Table '{POSTGRES_TABLE}' deleted from PostgreSQL.")
