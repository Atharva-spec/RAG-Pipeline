from dotenv import load_dotenv

from config.settings import RERANK_TOP_N, TOP_K
from generation.llm_client import get_answer
from generation.prompt_builder import build_prompt
from ingestion.chunker import chunk_documents
from ingestion.document_loader import load_documents
from retrieval.reranker import rerank_chunks
from retrieval.retriever import retrieve_chunks
from retrieval.vector_store import add_chunks, create_index, delete_index

load_dotenv()


def ingest(data_dir: str = "data/raw", rebuild: bool = False):
    """
    Load, chunk, embed, and store documents in OpenSearch using HNSW and bulk indexing.
    """
    if rebuild:
        print("--- Rebuilding OpenSearch index ---")
        delete_index()

    print("--- Starting Ingestion ---")
    documents = load_documents(data_dir)
    if not documents:
        print(f"No documents found in '{data_dir}'. Drop .pdf or .txt files there and re-run.")
        return

    print(f"Loaded {len(documents)} document(s)")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunk(s)")

    create_index()
    add_chunks(chunks)
    print("Ingestion complete. Chunks stored in OpenSearch.\n")


def query(question: str) -> str:
    raw = retrieve_chunks(question, k=TOP_K)
    if not raw:
        print("\nNo relevant context retrieved.")
        return "I could not find relevant context to answer your question."

    ranked = rerank_chunks(question, raw, top_n=RERANK_TOP_N)
    prompt = build_prompt(question, ranked)
    answer = get_answer(prompt)
    print(f"\nAnswer:\n{answer}\n")
    return answer


def chat_loop():
    """
    Interactive terminal chat session.
    """
    print("\n" + "=" * 50)
    print(" Native RAG Pipeline — Interactive Terminal Chat")
    print(" Type your question and press Enter.")
    print(" Type 'exit' or 'quit' to end session.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting chat. Goodbye!")
                break

            query(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Native RAG Pipeline - Production Core (Phase 1)")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe existing index and re-ingest all documents from scratch",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Ask a single question against your documents",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive chat session in the terminal",
    )
    args = parser.parse_args()

    if args.rebuild:
        ingest(rebuild=True)
    elif args.ingest:
        ingest(rebuild=False)
    elif args.query:
        query(args.query)
    elif args.chat:
        chat_loop()
    else:
        chat_loop()

