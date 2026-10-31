from dotenv import load_dotenv

from config.settings import RERANK_TOP_N, TOP_K
from generation.citation_resolver import resolve_citations
from generation.guardrails import (
    check_input_safety,
    detect_conversational_intent,
    safe_refusal_response,
    sanitize_output,
    validate_output_safety,
    validate_retrieval_confidence,
)
from generation.llm_client import get_answer
from generation.memory import ConversationMemory
from generation.prompt_builder import build_prompt
from generation.rephraser_agent import rephrase_query
from ingestion.auto_sync import folder_auto_sync
from retrieval.health_check import run_preflight_check
from retrieval.reranker import rerank_chunks
from retrieval.retriever import retrieve_chunks
from retrieval.vector_store import add_chunks, create_index, delete_index

load_dotenv()


def ingest(data_dir: str = "data/raw", rebuild: bool = False):
    """
    Ingests documents:
    - If rebuild=True: Wipes index and re-ingests everything from scratch.
    - If rebuild=False: Smart incremental sync (skips unchanged files in <5ms).
    """
    if rebuild:
        from ingestion.chunker import chunk_documents
        from ingestion.document_loader import load_documents

        print("\n--- Rebuilding PostgreSQL Vector Index from Scratch ---")
        delete_index()
        documents = load_documents(data_dir)
        if not documents:
            print(f"No documents found in '{data_dir}'. Drop .pdf or .txt files there and re-run.")
            return

        print(f"Loaded {len(documents)} document(s)")
        chunks = chunk_documents(documents)
        print(f"Created {len(chunks)} chunk(s)")

        create_index()
        add_chunks(chunks)
        print("Ingestion complete. All chunks stored in PostgreSQL.\n")
    else:
        from ingestion.incremental_ingest import sync_incremental
        sync_incremental(data_dir, verbose=True)


def query(
    question: str,
    memory: ConversationMemory | None = None,
) -> str:
    # 1. Input Safety Guardrail (Prompt injection & jailbreak detection)
    is_safe, refusal_reason = check_input_safety(question)
    if not is_safe:
        print(f"\n[Guardrail Alert] {refusal_reason}\n")
        return refusal_reason or "Input blocked by safety policy."

    # 2. Conversational Intent Detection (Chit-chat fast routing)
    intent_reply = detect_conversational_intent(question)
    if intent_reply is not None:
        if memory is not None:
            memory.add_turn(question, intent_reply)
        print(f"\nAnswer:\n{intent_reply}\n")
        return intent_reply

    # Automatically pick up files added to data/raw without requiring a
    # separate ingest command. The folder manager skips unchanged files.
    folder_auto_sync.ensure_latest()

    # 3. Rephraser Agent (Query reformulation, typo repair & context resolution)
    search_query = rephrase_query(question, memory=memory)

    # 4. High-recall hybrid retrieval. Keep both the rewritten query and the
    # user's exact wording so a rewrite can improve recall without becoming a
    # single point of failure.
    raw_chunks = retrieve_chunks(
        search_query,
        k=TOP_K,
        query_variants=[question],
    )
    if not raw_chunks:
        refusal = safe_refusal_response()
        if memory is not None:
            memory.add_turn(question, refusal)
        print(f"\nAnswer:\n{refusal}\n")
        return refusal

    # 5. Cross-Encoder Precision Reranking
    ranked_chunks = rerank_chunks(search_query, raw_chunks, top_n=RERANK_TOP_N)

    # 6. Zero-Hallucination Confidence Guardrail
    is_confident, qualified_chunks = validate_retrieval_confidence(ranked_chunks)
    if not is_confident or not qualified_chunks:
        refusal = safe_refusal_response()
        if memory is not None:
            memory.add_turn(question, refusal)
        print(f"\nAnswer:\n{refusal}\n")
        return refusal

    # 7. LLM Generation & Output Guardrail
    prompt = build_prompt(question, qualified_chunks, memory=memory)
    raw_answer = get_answer(prompt)
    is_safe_output, answer, alert_msg = validate_output_safety(raw_answer)
    if not is_safe_output and alert_msg:
        print(f"\n[Output Guardrail Alert] {alert_msg}")

    # 8. Verifiable Citation Resolution & Footer Attachment
    citations, citations_footer = resolve_citations(answer, qualified_chunks)
    full_output = f"{answer}{citations_footer}"

    # 9. Record Turn in Memory
    if memory is not None:
        source_names = [c["source"] for c in citations] if citations else [c.get("source", "Unknown") for c in qualified_chunks]
        memory.add_turn(question, full_output, sources=source_names)

    print(f"\nAnswer:\n{full_output}\n")
    return full_output


def chat_loop():
    """
    Interactive multi-turn terminal chat session with Conversational Memory, Health Check, and Guardrails.
    """
    # 1. Run Pre-flight Health Check
    run_preflight_check()
    folder_auto_sync.start()

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

    folder_auto_sync.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Native RAG Pipeline - Production Core (Phase 1)")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run smart incremental document ingestion (skips unchanged files in <5ms)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe existing index and re-ingest all documents from scratch",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Start live background folder watcher on data/raw/ for hands-free auto-indexing",
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
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run local RAGAS evaluation suite and generate benchmark report",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run automated cross-encoder confidence threshold calibration",
    )
    args = parser.parse_args()

    if args.calibrate:
        from evaluation.calibrate_threshold import run_confidence_calibration
        run_confidence_calibration()
    elif args.watch:
        from ingestion.watcher import start_file_watcher
        start_file_watcher()
    elif args.evaluate:
        from evaluation.run_eval import run_evaluation_cli
        run_evaluation_cli()
    elif args.rebuild:
        ingest(rebuild=True)
    elif args.ingest:
        ingest(rebuild=False)
    elif args.query:
        query(args.query)
    elif args.chat:
        chat_loop()
    else:
        chat_loop()

