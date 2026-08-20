from dotenv import load_dotenv

from config.settings import RERANK_TOP_N, TOP_K
from generation.guardrails import (
    check_input_safety,
    detect_conversational_intent,
    safe_refusal_response,
    sanitize_output,
    validate_retrieval_confidence,
)
from generation.llm_client import get_answer
from generation.memory import ConversationMemory
from generation.prompt_builder import build_prompt
from generation.query_condenser import condense_query
from ingestion.chunker import chunk_documents
from ingestion.document_loader import load_documents
from retrieval.health_check import run_preflight_check
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


def query(

    question: str,
    memory: ConversationMemory | None = None,
) -> str:
    # 1. Input Safety Guardrail (Prompt injection & jailbreak detection)
    is_safe, refusal_reason = check_input_safety(question)
    if not is_safe:
        print(f"\n[Guardrail Alert] {refusal_reason}\n")
        return refusal_reason or "Input blocked by safety policy."

    # 2. Conversational Intent Guardrail (Chit-chat / Greetings 0ms fast-path)
    chitchat_response = detect_conversational_intent(question)
    if chitchat_response:
        if memory is not None:
            memory.add_turn(question, chitchat_response)
        print(f"\nAnswer:\n{chitchat_response}\n")
        return chitchat_response

    # 3. Query Condensation (Resolves pronouns and follow-up context)
    search_query = condense_query(question, memory=memory)

    # 4. Retrieval (executed on the condensed standalone query)
    raw = retrieve_chunks(search_query, k=TOP_K)
    if not raw:
        refusal = safe_refusal_response()
        if memory is not None:
            memory.add_turn(question, refusal)
        print(f"\nAnswer:\n{refusal}\n")
        return refusal

    # 5. Reranking & Confidence Scoring
    ranked = rerank_chunks(search_query, raw, top_n=RERANK_TOP_N)

    # 6. Retrieval Confidence Guardrail (Zero-Hallucination Gate)
    is_confident, qualified_chunks = validate_retrieval_confidence(ranked)
    if not is_confident or not qualified_chunks:
        refusal = safe_refusal_response()
        if memory is not None:
            memory.add_turn(question, refusal)
        print(f"\nAnswer:\n{refusal}\n")
        return refusal

    # 7. LLM Generation & Output Sanitization
    prompt = build_prompt(question, qualified_chunks, memory=memory)
    raw_answer = get_answer(prompt)
    answer = sanitize_output(raw_answer)

    # 8. Record Turn in Memory
    if memory is not None:
        sources = list({c.get("source", "Unknown") for c in qualified_chunks})
        memory.add_turn(question, answer, sources=sources)

    print(f"\nAnswer:\n{answer}\n")
    return answer


def chat_loop():
    """
    Interactive multi-turn terminal chat session with Conversational Memory, Health Check, and Guardrails.
    """
    # 1. Run Pre-flight Health Check
    run_preflight_check()

    memory = ConversationMemory(max_turns=5)

    print("\n" + "=" * 55)
    print(" Native RAG Pipeline — Conversational Terminal Chat")
    print(" • Type your question and press Enter.")
    print(" • Type '/clear' to reset chat memory.")
    print(" • Type 'exit' or 'quit' to end session.")
    print("=" * 55 + "\n")


    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting chat. Goodbye!")
                break

            if user_input.lower() in ("/clear", "/reset", "clear"):
                memory.clear()
                print("  Memory cleared. Starting a fresh conversation.")
                continue

            query(user_input, memory=memory)
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

