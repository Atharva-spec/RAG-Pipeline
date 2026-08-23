import psycopg

from config.settings import HYBRID_SEARCH_ENABLED, POSTGRES_PORT, TOP_K
from retrieval.vector_store import search


def retrieve_chunks(query: str, k: int = TOP_K) -> list[dict]:
    mode_str = "Hybrid (Vector + GIN Full-Text with RRF)" if HYBRID_SEARCH_ENABLED else "Dense Vector"
    try:
        print(f"  Searching PostgreSQL [{mode_str}]: '{query}'")
        results = search(query, k=k)
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