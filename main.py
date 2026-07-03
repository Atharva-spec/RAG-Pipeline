from dotenv import load_dotenv

from config.settings import TOP_K, RERANK_TOP_N
from generation.llm_client import get_answer
from generation.prompt_builder import build_prompt
from ingestion.chunker import chunk_documents
from ingestion.document_loader import load_documents
from retrieval.reranker import rerank_chunks
from retrieval.retriever import retrieve_chunks
from retrieval.vector_store import add_chunks, create_index

load_dotenv()


def ingest(data_dir: str = "data/raw"):
    """
    Run once to load, chunk, embed and store your documents.
    Re-run whenever you add new documents.
    """
    print("--- Starting ingestion ---")

    documents = load_documents(data_dir)
    print(f"Loaded {len(documents)} document(s)")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunk(s)")

    create_index()
    add_chunks(chunks)

    print("Ingestion complete. Chunks stored in OpenSearch.\n")


def query(question: str) -> str:
    raw = retrieve_chunks(question, k=TOP_K)
    ranked = rerank_chunks(question, raw, top_n=RERANK_TOP_N)
    prompt = build_prompt(question, ranked)
    answer = get_answer(prompt)
    print(f"\nAnswer: {answer}")
    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Native RAG Pipeline")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion (do this first)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Ask a question against your documents",
    )
    args = parser.parse_args()

    if args.ingest:
        ingest()
    elif args.query:
        query(args.query)
    else:
        print("Usage:")
        print("  python main.py --ingest")
        print("  python main.py --query 'Your question here'")
