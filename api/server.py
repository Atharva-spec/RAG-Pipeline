import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import (
    API_HOST,
    API_PORT,
    AUTO_SYNC_ENABLED,
    DATA_RAW_DIR,
    GROQ_MODEL,
    GROQ_STT_MODEL,
    GUARDRAIL_CONFIDENCE_THRESHOLD,
    LLM_BACKEND,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RERANK_TOP_N,
    STT_BACKEND,
    STT_DEFAULT_LANGUAGE,
    STT_ENABLED,
    STT_MAX_FILE_SIZE_BYTES,
    TOP_K,
)
from core.cache import (
    clear_cached_session,
    get_cache_stats,
    invalidate_cached_chunks,
    is_redis_available,
)
from generation.citation_resolver import resolve_citations
from generation.guardrails import (
    check_input_safety,
    detect_conversational_intent,
    safe_refusal_response,
    validate_output_safety,
    validate_retrieval_confidence,
)
from generation.llm_client import get_answer, stream_answer, transcribe_audio
from generation.memory import ConversationMemory
from generation.prompt_builder import build_prompt
from generation.query_condenser import condense_query
from generation.rephraser_agent import rephrase_query
from ingestion.document_loader import SUPPORTED_EXTENSIONS
from ingestion.auto_sync import folder_auto_sync
from retrieval.health_check import check_llm, check_postgres
from retrieval.reranker import rerank_chunks
from retrieval.retriever import retrieve_chunks
from retrieval.vector_store import get_connection

logger = logging.getLogger("rag_api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run automatic raw-folder synchronization while the API is alive."""
    if AUTO_SYNC_ENABLED:
        folder_auto_sync.start()
    try:
        yield
    finally:
        if AUTO_SYNC_ENABLED:
            folder_auto_sync.stop()

app = FastAPI(
    title="Enterprise Conversational RAG Knowledge Engine API",
    description="High-performance REST & SSE Token Streaming API with PostgreSQL pgvector and Redis Caching",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for Next.js, Stitch previews, and mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# Pydantic Request & Response Models
# =========================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., description="User query or follow-up question")
    session_id: Optional[str] = Field("default", description="Session ID for conversational memory")
    use_cache: bool = Field(True, description="Whether to check/populate Tier-1 Redis answer cache")
    selected_docs: Optional[List[str]] = Field(None, description="Optional list of filenames to filter scope")


class TelemetryData(BaseModel):
    latency_ms: float
    retrieved_chunks: int
    reranked_top_n: int
    top_confidence_score: float
    llm_backend: str
    model: str


class CitationItem(BaseModel):
    index: int
    chunk_id: str
    source: str
    page: Optional[int] = None
    confidence_score: float
    exact_quote: str
    parent_context: str


class QueryResponse(BaseModel):
    answer: str
    full_output: str
    citations: List[CitationItem]
    confidence_score: float
    telemetry: TelemetryData
    cached: bool = False
    refusal: bool = False


class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    model: str
    latency_ms: float


class DocumentItem(BaseModel):
    id: str
    filename: str
    extension: str
    size_bytes: int
    total_chunks: int
    sha256_hash: str
    uploaded_at: str


class EngineConfig(BaseModel):
    model: str = GROQ_MODEL
    temperature: float = LLM_TEMPERATURE
    confidence_threshold: float = GUARDRAIL_CONFIDENCE_THRESHOLD
    top_k: int = TOP_K
    rerank_top_n: int = RERANK_TOP_N
    use_hierarchical_parents: bool = True
    guardrails_enabled: bool = True


# Global runtime configuration state
_runtime_config = EngineConfig()


# =========================================================================
# System Health & Status Endpoints
# =========================================================================

@app.get("/v1/health")
def get_health_status():
    """
    Returns comprehensive pre-flight status for PostgreSQL, Groq Cloud, and Redis.
    """
    pg_ok, pg_msg = check_postgres()
    llm_ok, llm_msg = check_llm()
    redis_stats = get_cache_stats()

    all_healthy = pg_ok and llm_ok

    return {
        "status": "healthy" if all_healthy else "degraded",
        "database": {
            "online": pg_ok,
            "message": pg_msg,
            "engine": "PostgreSQL (pgvector HNSW + GIN tsvector)",
        },
        "llm_engine": {
            "online": llm_ok,
            "message": llm_msg,
            "backend": LLM_BACKEND.upper(),
            "model": _runtime_config.model,
        },
        "redis_cache": redis_stats,
        "automatic_sync": folder_auto_sync.status(),
        "speech_to_text": {
            "enabled": STT_ENABLED,
            "backend": STT_BACKEND,
            "model": GROQ_STT_MODEL,
            "max_file_size_bytes": STT_MAX_FILE_SIZE_BYTES,
        },
        "timestamp": time.time(),
    }


@app.get("/v1/cache/stats")
def get_cache_diagnostics():
    """
    Returns Redis cache statistics and memory metrics.
    """
    return get_cache_stats()


# =========================================================================
# Query & Streaming Endpoints
# =========================================================================

@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe_query_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None,
):
    """Transcribe a browser recording for use as a RAG query."""
    if not STT_ENABLED:
        raise HTTPException(status_code=503, detail="Speech-to-text is disabled.")

    filename = os.path.basename(file.filename or "recording.webm")
    extension = os.path.splitext(filename)[1].lower()
    supported_audio_extensions = {
        ".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"
    }
    if extension not in supported_audio_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio type. Use webm, wav, mp3, m4a, ogg, flac, or mp4.",
        )

    audio_bytes = await file.read(STT_MAX_FILE_SIZE_BYTES + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio recording is empty.")
    if len(audio_bytes) > STT_MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio recording exceeds the {STT_MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    started_at = time.time()
    try:
        result = transcribe_audio(
            audio_bytes,
            filename,
            content_type=file.content_type,
            language=language or STT_DEFAULT_LANGUAGE or None,
        )
    except ValueError as exc:
        logger.error("Speech-to-text configuration error: %s", exc)
        raise HTTPException(status_code=503, detail="Speech-to-text is not configured.") from exc
    except Exception as exc:
        logger.error("Speech-to-text request failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Speech-to-text provider failed.") from exc

    return TranscriptionResponse(
        text=result["text"],
        language=result.get("language"),
        model=result["model"],
        latency_ms=round((time.time() - started_at) * 1000, 2),
    )

@app.post("/v1/query", response_model=QueryResponse)
def execute_query(payload: QueryRequest):
    """
    Executes end-to-end RAG pipeline with Tier-1 Redis answer caching.
    """
    t0 = time.time()
    question = payload.question.strip()
    session_id = payload.session_id or "default"

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Input Safety Guardrail
    is_safe, refusal_reason = check_input_safety(question)
    if not is_safe:
        return QueryResponse(
            answer=refusal_reason or "Input blocked by safety policy.",
            full_output=refusal_reason or "Input blocked by safety policy.",
            citations=[],
            confidence_score=0.0,
            telemetry=TelemetryData(
                latency_ms=round((time.time() - t0) * 1000, 2),
                retrieved_chunks=0,
                reranked_top_n=0,
                top_confidence_score=0.0,
                llm_backend=LLM_BACKEND,
                model=_runtime_config.model,
            ),
            refusal=True,
        )

    # 2. Conversational Intent Detection (Chit-Chat)
    intent_reply = detect_conversational_intent(question)
    if intent_reply is not None:
        memory = ConversationMemory(session_id=session_id)
        memory.add_turn(question, intent_reply)
        return QueryResponse(
            answer=intent_reply,
            full_output=intent_reply,
            citations=[],
            confidence_score=1.0,
            telemetry=TelemetryData(
                latency_ms=round((time.time() - t0) * 1000, 2),
                retrieved_chunks=0,
                reranked_top_n=0,
                top_confidence_score=1.0,
                llm_backend="intent_fast_router",
                model="heuristic",
            ),
            refusal=False,
        )

    # Catch up synchronously if a document was dropped immediately before the
    # query. The background watcher handles normal changes between requests.
    folder_auto_sync.ensure_latest()

    # 3. Rephraser Agent (Query reformulation, typo repair & context resolution)
    memory = ConversationMemory(session_id=session_id)
    search_query = rephrase_query(question, memory=memory)

    # 4. High-recall hybrid retrieval. The original question remains in the
    # candidate set, and document scoping is pushed into the database search.
    raw_chunks = retrieve_chunks(
        search_query,
        k=_runtime_config.top_k,
        query_variants=[question],
        selected_docs=payload.selected_docs,
    )

    if not raw_chunks:
        refusal = safe_refusal_response()
        memory.add_turn(question, refusal)
        return QueryResponse(
            answer=refusal,
            full_output=refusal,
            citations=[],
            confidence_score=0.0,
            telemetry=TelemetryData(
                latency_ms=round((time.time() - t0) * 1000, 2),
                retrieved_chunks=0,
                reranked_top_n=0,
                top_confidence_score=0.0,
                llm_backend=LLM_BACKEND,
                model=_runtime_config.model,
            ),
            refusal=True,
        )

    # 5. Cross-Encoder Precision Reranking
    ranked_chunks = rerank_chunks(search_query, raw_chunks, top_n=_runtime_config.rerank_top_n)

    # 6. Zero-Hallucination Confidence Guardrail
    is_confident, qualified_chunks = validate_retrieval_confidence(
        ranked_chunks,
        threshold=_runtime_config.confidence_threshold,
    )
    if not is_confident or not qualified_chunks:
        refusal = safe_refusal_response()
        memory.add_turn(question, refusal)
        top_score = ranked_chunks[0].get("confidence_score", 0.0) if ranked_chunks else 0.0
        return QueryResponse(
            answer=refusal,
            full_output=refusal,
            citations=[],
            confidence_score=top_score,
            telemetry=TelemetryData(
                latency_ms=round((time.time() - t0) * 1000, 2),
                retrieved_chunks=len(raw_chunks),
                reranked_top_n=len(ranked_chunks),
                top_confidence_score=top_score,
                llm_backend=LLM_BACKEND,
                model=_runtime_config.model,
            ),
            refusal=True,
        )

    # 7. LLM Generation & Output Guardrail
    prompt = build_prompt(question, qualified_chunks, memory=memory)
    raw_answer = get_answer(prompt, temperature=_runtime_config.temperature)
    is_safe_output, answer, _ = validate_output_safety(raw_answer)

    # 8. Verifiable Citation Resolution
    citations, citations_footer = resolve_citations(answer, qualified_chunks)
    full_output = f"{answer}{citations_footer}"

    # 9. Record Turn in Memory
    source_names = [c["source"] for c in citations] if citations else [c.get("source", "Unknown") for c in qualified_chunks]
    memory.add_turn(question, full_output, sources=source_names)

    top_confidence = qualified_chunks[0].get("confidence_score", 0.95)
    latency_ms = round((time.time() - t0) * 1000, 2)

    citation_items = [
        CitationItem(
            index=c.get("index", idx + 1),
            chunk_id=c.get("chunk_id", f"c_{idx}"),
            source=c.get("source", "Unknown"),
            page=c.get("page"),
            confidence_score=c.get("confidence_score", top_confidence),
            exact_quote=c.get("exact_quote") or c.get("excerpt") or c.get("quote", ""),
            parent_context=c.get("parent_context") or c.get("text", ""),
        )
        for idx, c in enumerate(citations or qualified_chunks)
    ]

    response_payload = QueryResponse(
        answer=answer,
        full_output=full_output,
        citations=citation_items,
        confidence_score=top_confidence,
        telemetry=TelemetryData(
            latency_ms=latency_ms,
            retrieved_chunks=len(raw_chunks),
            reranked_top_n=len(qualified_chunks),
            top_confidence_score=top_confidence,
            llm_backend=LLM_BACKEND,
            model=_runtime_config.model,
        ),
        cached=False,
        refusal=False,
    )

    return response_payload



@app.post("/v1/query/stream")
def execute_query_stream(payload: QueryRequest):
    """
    Streams generated response tokens and citations via Server-Sent Events (SSE).
    """
    question = payload.question.strip()
    session_id = payload.session_id or "default"

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def sse_event_generator():
        t0 = time.time()

        # 1. Input Safety Guardrail
        is_safe, refusal_reason = check_input_safety(question)
        if not is_safe:
            refusal_text = refusal_reason or "Input blocked by safety policy."
            yield f"data: {json.dumps({'event': 'token', 'content': refusal_text})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'refusal': True})}\n\n"
            return

        # 2. Chit-Chat Intent Detection
        intent_reply = detect_conversational_intent(question)
        if intent_reply is not None:
            memory = ConversationMemory(session_id=session_id)
            memory.add_turn(question, intent_reply)
            yield f"data: {json.dumps({'event': 'token', 'content': intent_reply})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'refusal': False})}\n\n"
            return

        # Ensure a file added just before an SSE request is searchable now;
        # the background watcher covers later changes.
        folder_auto_sync.ensure_latest()

        # 3. Rephraser Agent & Retrieval
        memory = ConversationMemory(session_id=session_id)
        search_query = rephrase_query(question, memory=memory)

        raw_chunks = retrieve_chunks(
            search_query,
            k=_runtime_config.top_k,
            query_variants=[question],
            selected_docs=payload.selected_docs,
        )

        if not raw_chunks:
            refusal = safe_refusal_response()
            memory.add_turn(question, refusal)
            yield f"data: {json.dumps({'event': 'token', 'content': refusal})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'refusal': True})}\n\n"
            return

        # 4. Reranking & Confidence Gate
        ranked_chunks = rerank_chunks(search_query, raw_chunks, top_n=_runtime_config.rerank_top_n)
        is_confident, qualified_chunks = validate_retrieval_confidence(
            ranked_chunks,
            threshold=_runtime_config.confidence_threshold,
        )

        top_confidence = qualified_chunks[0].get("confidence_score", 0.0) if qualified_chunks else 0.0

        # Send Telemetry Event
        telemetry = {
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "retrieved_chunks": len(raw_chunks),
            "reranked_top_n": len(qualified_chunks),
            "top_confidence_score": top_confidence,
            "llm_backend": LLM_BACKEND,
            "model": _runtime_config.model,
        }
        yield f"data: {json.dumps({'event': 'telemetry', 'data': telemetry})}\n\n"

        if not is_confident or not qualified_chunks:
            refusal = safe_refusal_response()
            memory.add_turn(question, refusal)
            yield f"data: {json.dumps({'event': 'token', 'content': refusal})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'refusal': True})}\n\n"
            return

        # 5. Token-by-Token Streaming Generation
        prompt = build_prompt(question, qualified_chunks, memory=memory)
        accumulated_text = ""

        try:
            for token in stream_answer(prompt, temperature=_runtime_config.temperature):
                accumulated_text += token
                yield f"data: {json.dumps({'event': 'token', 'content': token})}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
            return

        # 6. Citation Attachment
        is_safe_output, clean_answer, _ = validate_output_safety(accumulated_text)
        citations, citations_footer = resolve_citations(clean_answer, qualified_chunks)

        full_output = f"{clean_answer}{citations_footer}"
        source_names = [c["source"] for c in citations] if citations else [c.get("source", "Unknown") for c in qualified_chunks]
        memory.add_turn(question, full_output, sources=source_names)

        citation_items = [
            {
                "index": c.get("index", idx + 1),
                "chunk_id": c.get("chunk_id", f"c_{idx}"),
                "source": c.get("source", "Unknown"),
                "page": c.get("page"),
                "confidence_score": c.get("confidence_score", top_confidence),
                "exact_quote": c.get("exact_quote") or c.get("excerpt") or c.get("quote", ""),
                "parent_context": c.get("parent_context") or c.get("text", ""),
            }
            for idx, c in enumerate(citations or qualified_chunks)
        ]

        yield f"data: {json.dumps({'event': 'citations', 'citations': citation_items, 'footer': citations_footer})}\n\n"
        yield f"data: {json.dumps({'event': 'done', 'refusal': False})}\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")


# =========================================================================
# Document Management Endpoints (Knowledge Vault)
# =========================================================================

@app.get("/v1/documents", response_model=List[DocumentItem])
def get_indexed_documents():
    """
    Returns inventory of indexed documents in the Knowledge Vault.
    """
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    doc_items = []

    # Get chunk counts from PostgreSQL
    chunk_counts_by_source: Dict[str, int] = {}
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source, COUNT(*) FROM document_chunks GROUP BY source;"
                )
                for row in cur.fetchall():
                    chunk_counts_by_source[row["source"]] = row["count"]
    except Exception as e:
        logger.debug(f"Could not query chunk counts from DB: {e}")

    for filename in sorted(os.listdir(DATA_RAW_DIR)):
        filepath = os.path.join(DATA_RAW_DIR, filename)
        if not os.path.isfile(filepath) or filename.startswith("."):
            continue

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
        size_bytes = os.path.getsize(filepath)
        mtime = os.path.getmtime(filepath)
        uploaded_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

        with open(filepath, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        doc_items.append(
            DocumentItem(
                id=f"doc_{file_hash}",
                filename=filename,
                extension=ext,
                size_bytes=size_bytes,
                total_chunks=chunk_counts_by_source.get(filename, 0),
                sha256_hash=file_hash,
                uploaded_at=uploaded_at,
            )
        )

    return doc_items


@app.post("/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document to data/raw/ and triggers incremental indexing into PostgreSQL.
    """
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    # Never allow a client-provided path to escape data/raw.
    filename = os.path.basename(file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    extension = os.path.splitext(filename)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type '{extension or '[none]'}'. Supported types: {supported}",
        )

    filepath = os.path.join(DATA_RAW_DIR, filename)
    temporary_path = f"{filepath}.uploading"

    # Write completely before making the file visible to the folder watcher or
    # incremental synchronizer.
    with open(temporary_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    os.replace(temporary_path, filepath)

    # Trigger smart incremental sync
    try:
        sync_res = folder_auto_sync.sync_now()
        chunks_added = sync_res.get("chunks_added", 0)
    except Exception as e:
        logger.error(f"Error during document indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document upload failed during indexing: {e}")

    return {
        "status": "success",
        "message": f"Document '{filename}' uploaded and indexed successfully ({chunks_added} chunks).",
        "filename": filename,
        "size_bytes": os.path.getsize(filepath),
        "chunks_indexed": chunks_added,
    }


@app.delete("/v1/documents/{filename}")
def delete_document(filename: str):
    """
    Deletes a document from disk, purges its chunks from PostgreSQL,
    and invalidates its chunk cache in Redis.
    """
    filepath = os.path.join(DATA_RAW_DIR, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

    # Purge from PostgreSQL and invalidate Redis chunk cache
    try:
        from ingestion.incremental_ingest import delete_document_chunks
        deleted_count = delete_document_chunks(filename)
    except Exception as e:
        logger.error(f"Error purging chunks for {filename}: {e}")
        deleted_count = 0

    return {
        "status": "success",
        "message": f"Document '{filename}' and {deleted_count} indexed chunks have been deleted.",
    }



# =========================================================================
# Model & Runtime Configuration Endpoints
# =========================================================================

@app.get("/v1/models")
def get_available_models():
    """
    Returns available Cloud LLM models and active configuration.
    """
    return {
        "active_model": _runtime_config.model,
        "available_models": [
            {
                "id": "openai/gpt-oss-120b",
                "name": "Groq GPT-OSS 120B",
                "provider": "Groq LPU Cloud",
                "context_window": "128k",
                "speed": "~800 tok/sec",
                "default": True,
            },
            {
                "id": "llama-3.3-70b-versatile",
                "name": "Llama 3.3 70B Versatile",
                "provider": "Groq LPU Cloud",
                "context_window": "128k",
                "speed": "~350 tok/sec",
                "default": False,
            },
            {
                "id": "llama-3.1-8b-instant",
                "name": "Llama 3.1 8B Instant",
                "provider": "Groq LPU Cloud",
                "context_window": "128k",
                "speed": "~1,000 tok/sec",
                "default": False,
            },
        ],
        "configuration": _runtime_config.model_dump(),
    }


@app.post("/v1/models/config")
def update_engine_config(config: EngineConfig):
    """
    Dynamically updates runtime inference parameters.
    """
    global _runtime_config
    _runtime_config = config
    return {
        "status": "success",
        "message": "Engine configuration updated.",
        "configuration": _runtime_config.model_dump(),
    }


@app.post("/v1/memory/clear")
def clear_session_memory(session_id: str = "default"):
    """
    Clears conversation memory for a session ID from RAM and Redis.
    """
    clear_cached_session(session_id)
    return {
        "status": "success",
        "message": f"Memory for session '{session_id}' cleared.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
