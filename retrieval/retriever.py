from retrieval.vector_store import search
from config.settings import TOP_K
from opensearchpy.exceptions import NotFoundError

def retrieve_chunks(query: str, k: int = TOP_K) -> list[dict]:
    try:
        print(f"  Querying OpenSearch: '{query}'")
        results = search(query, k=k)
        print(f"  Retrieved {len(results)} chunk(s)")
        return results
    except NotFoundError:
        print("  ERROR: No index found. Run --ingest first.")
        return []