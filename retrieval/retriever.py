from opensearchpy.exceptions import ConnectionError, NotFoundError

from config.settings import HYBRID_SEARCH_ENABLED, TOP_K
from retrieval.vector_store import search


def retrieve_chunks(query: str, k: int = TOP_K) -> list[dict]:
    mode_str = "Hybrid (Vector + BM25 with RRF)" if HYBRID_SEARCH_ENABLED else "Dense Vector"
    try:
        print(f"  Searching OpenSearch [{mode_str}]: '{query}'")
        results = search(query, k=k)
        print(f"  Retrieved {len(results)} candidate chunk(s)")
        return results
    except NotFoundError:
        print("  ERROR: Index does not exist. Please run with --ingest first.")
        return []
    except ConnectionError:
        print("  ERROR: Could not connect to OpenSearch. Ensure the container is running on port 9200.")
        return []
    except Exception as e:
        print(f"  ERROR during retrieval: {e}")
        return []