from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import psycopg

from config.settings import HYBRID_SEARCH_ENABLED, POSTGRES_PORT, RRF_K, TOP_K
from retrieval.vector_store import search


def retrieve_chunks(
    query: str,
    k: int = TOP_K,
    query_variants: Optional[Sequence[str]] = None,
    selected_docs: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Retrieve high-recall candidates using one or more equivalent queries.

    The original question is intentionally allowed alongside a rewritten query:
    query rewriting is useful for typos and follow-ups, but a rewrite should
    never be able to erase a term that exists verbatim in the document.
    ``selected_docs`` is applied inside PostgreSQL before ranking so a scoped
    query cannot be crowded out by unrelated documents.
    """
    mode_str = "Hybrid (Vector + GIN Full-Text with RRF)" if HYBRID_SEARCH_ENABLED else "Dense Vector"
    candidates = [query, *(query_variants or [])]
    queries = []
    seen = set()
    for candidate in candidates:
        clean = (candidate or "").strip()
        key = clean.casefold()
        if clean and key not in seen:
            queries.append(clean)
            seen.add(key)

    if not queries:
        return []

    try:
        print(f"  Searching PostgreSQL [{mode_str}] with {len(queries)} query variant(s)")
        fetch_k = max(k * 3, 24)

        def run(candidate: str) -> list[dict]:
            # Keep the legacy call shape when no scope is requested; this also
            # makes the dispatcher easy to replace in lightweight deployments.
            if selected_docs:
                return search(candidate, k=fetch_k, sources=selected_docs)
            return search(candidate, k=fetch_k)

        with ThreadPoolExecutor(max_workers=min(3, len(queries))) as executor:
            result_sets = list(executor.map(run, queries))

        # Fuse the variant result lists. This is deliberately rank-based so a
        # lexical exact match and a semantically similar match both contribute.
        fused_scores: dict[str, float] = {}
        lookup: dict[str, dict] = {}
        for result_set in result_sets:
            for rank, chunk in enumerate(result_set, start=1):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if not chunk_id:
                    continue
                lookup.setdefault(chunk_id, chunk)
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1.0 / (RRF_K + rank))

        results = []
        for chunk_id in sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]:
            item = dict(lookup[chunk_id])
            item["retrieval_score"] = item.get("score", 0.0)
            item["score"] = round(fused_scores[chunk_id], 6)
            item["method"] = "multi_query_rrf" if len(queries) > 1 else item.get("method", "hybrid")
            results.append(item)

        print(f"  Retrieved {len(results)} candidate chunk(s)")
        return results
    except psycopg.OperationalError:
        print(f"  ERROR: Could not connect to PostgreSQL. Ensure the container is running on port {POSTGRES_PORT}.")
        return []
    except psycopg.ProgrammingError as e:
        print(f"  ERROR: Database or table does not exist ({e}). Please run 'npm run ingest' first.")
        return []
    except Exception as e:
        print(f"  ERROR during retrieval: {e}")
        return []
