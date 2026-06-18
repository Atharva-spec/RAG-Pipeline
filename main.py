import os
from dotenv import load_dotenv

from ingestion.document_loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import get_embedding_model
from retrieval.vector_store import VectorStore
from retrieval.retriever import retrieve_chunks
from retrieval.reranker import rerank_chunks
from generation.prompt_builder import build_prompt
from generation.llm_client import get_answer
from config.settings import TOP_K, RERANK_TOP_N

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

    embedding_model = get_embedding_model()
    store = VectorStore(embedding_model)
    store.add_chunks(chunks)
    store.persist()

    print("Ingestion complete. Vector DB saved.\n")


def query(user_question: str) -> str:
    """
    Full query pipeline:
    question → retrieve → rerank → prompt → LLM → answer
    """
    print(f"\nQuestion: {user_question}")
    print("---")

    embedding_model = get_embedding_model()
    store = VectorStore(embedding_model)
    store.load()

    raw_chunks = retrieve_chunks(store, user_question, k=TOP_K)
    print(f"Retrieved {len(raw_chunks)} chunk(s) from vector DB")

    ranked_chunks = rerank_chunks(user_question, raw_chunks, top_n=RERANK_TOP_N)
    print(f"Reranked down to top {len(ranked_chunks)} chunk(s)")

    prompt = build_prompt(user_question, ranked_chunks)

    answer = get_answer(prompt)
    print(f"\nAnswer: {answer}\n")

    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Native RAG Pipeline")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion (do this first)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Ask a question against your documents"
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