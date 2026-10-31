# Conversational Hybrid RAG Pipeline

An end-to-end document question-answering engine built around PostgreSQL `pgvector`, PostgreSQL full-text search, hierarchical chunking, cross-encoder reranking, guardrails, citations, Redis-backed session memory, and a FastAPI API.

The repository contains a usable local prototype and service API. It is not yet a hardened multi-tenant production service: authentication, tenant isolation, upload hardening, rate limiting, production observability, and deployment controls still need to be added before exposing it to untrusted users.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Database: PostgreSQL pgvector](https://img.shields.io/badge/database-PostgreSQL%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![LLM: Groq](https://img.shields.io/badge/LLM-Groq-f55036.svg)](https://groq.com/)

## What the pipeline does

```text
Documents → parse and normalize → hierarchical chunks → embeddings
         → PostgreSQL hybrid retrieval → cross-encoder reranking
         → confidence gate → grounded LLM answer → citations and telemetry
```

### Retrieval and generation flow

1. **Input guardrails** check prompt injection, jailbreak patterns, encoded payloads, homoglyphs, and leetspeak.
2. **Intent routing** answers simple greetings and thanks without touching the vector store.
3. **Query normalization** repairs common typos and rewrites conversational questions for retrieval.
4. **Hybrid retrieval** runs dense HNSW vector search and PostgreSQL GIN full-text search in parallel, then combines ranks with reciprocal rank fusion.
5. **Reranking** uses `cross-encoder/ms-marco-MiniLM-L-6-v2`, preferably through ONNX Runtime, with a PyTorch fallback.
6. **Confidence gating** refuses answers when the reranker score is below the configured threshold. The score is a heuristic gate and should be calibrated against each customer corpus before being treated as a probability.
7. **Context construction** resolves parent chunks, removes duplicate parent blocks, and sanitizes document text against indirect prompt injection.
8. **Generation** calls the configured Groq-compatible OpenAI client.
9. **Output protection** redacts selected secrets and system-prompt leakage, then attaches structured citations and a sources footer.

## Current capabilities

- PostgreSQL `pgvector` HNSW cosine search with a generated English `tsvector` column and GIN index.
- Reciprocal rank fusion over vector and lexical search results.
- Hierarchical parent/child chunks by default: 400-character children, 2,000-character parents, with overlap.
- Alternative semantic-section and recursive chunking strategies.
- PDF, DOC/DOCX, PPT/PPTX, TXT, Markdown, HTML, and HTM parsing through Unstructured.
- PDF text extraction fallback through `pypdf` when Unstructured fails.
- SHA-256 incremental ingestion that skips unchanged files and purges deleted or modified sources.
- Optional polling watcher for automatic folder synchronization.
- ONNX reranker export and lazy model initialization, with PyTorch fallback if ONNX setup fails.
- Five guardrail layers covering input, conversational routing, document-content sanitization, retrieval confidence, and output filtering.
- Redis-backed 24-hour conversation memory for explicitly named sessions.
- FastAPI JSON and Server-Sent Events query endpoints.
- Optional speech-to-text query input through browser microphone recordings and Groq Whisper transcription.
- Document inventory, upload, delete, health, cache diagnostics, model listing/configuration, and memory-clear endpoints.
- Local evaluation metrics for faithfulness, answer relevance, context precision, and context recall.
- Confidence-threshold calibration utility.

## Requirements

- Python 3.10 or newer.
- Node.js and npm for the cross-platform command wrapper.
- Docker Desktop or another Docker Engine for PostgreSQL and Redis.
- A Groq API key for generated answers.
- Network access on first run if the embedding and reranker models are not already cached locally.

## Quick start

### 1. Install the environment

```bash
git clone https://github.com/Atharva-spec/RAG-Pipeline.git
cd RAG-pipeline
npm install
```

`npm install` creates or reuses the project virtual environment and installs `requirements.txt`.

### 2. Configure `.env`

Copy `.env.example` to `.env` and set at least the database and LLM values:

```dotenv
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=change-this-password

REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_ENABLED=true

AUTO_SYNC_ENABLED=true
AUTO_SYNC_INTERVAL_SECONDS=2.0

LLM_BACKEND=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

The supplied Docker Compose file maps Redis container port `6379` to host port `6380`. PostgreSQL is mapped to host port `5432`.

### 3. Start dependencies and ingest documents

Place supported files in `data/raw/`, then run:

```bash
npm run docker:up
npm run ingest
npm start
```

The first ingestion may download embedding and reranker models and can take several minutes. The query path cannot answer from a document that has not been successfully indexed.

When the API or interactive CLI is running, supported files added to `data/raw/` are detected automatically. The API runs a background folder watcher and also performs a change-aware catch-up immediately before document queries, so a newly copied file becomes searchable without manually running `npm run ingest`.

### 4. Start the API

```bash
npm run server
```

The development server listens on `http://localhost:8000`. OpenAPI documentation is available at `http://localhost:8000/docs`.

For a production deployment, run behind an authenticated reverse proxy or add application authentication before exposing any endpoint. The current API does not provide authentication or tenant isolation.

## CLI commands

| Purpose | npm command | macOS/Linux | Windows |
| --- | --- | --- | --- |
| Interactive chat | `npm start` | `./rag --chat` | `rag.bat --chat` |
| One-off query | `npm run query -- "question"` | `./rag --query "question"` | `rag.bat --query "question"` |
| Incremental ingestion | `npm run ingest` | `./rag --ingest` | `rag.bat --ingest` |
| Rebuild the index | `npm run rebuild` | `./rag --rebuild` | `rag.bat --rebuild` |
| Watch `data/raw/` | `npm run watch` | `./rag --watch` | `rag.bat --watch` |
| Run evaluation | `npm run eval` | `./rag --evaluate` | `rag.bat --evaluate` |
| Calibrate threshold | `npm run calibrate` | `./rag --calibrate` | `rag.bat --calibrate` |
| Run tests | `npm test` | `npm test` | `npm test` |
| Start PostgreSQL and Redis | `npm run docker:up` | `docker compose up -d` | `docker compose up -d` |
| Stop PostgreSQL and Redis | `npm run docker:down` | `docker compose down` | `docker compose down` |

Interactive chat commands:

- `/clear`, `/reset`, or `clear`: clear the current chat memory.
- `exit`, `quit`, or `q`: leave the chat.

## HTTP API

All API routes are under `/v1`.

| Method and route | Purpose |
| --- | --- |
| `GET /v1/health` | PostgreSQL, Groq client, Redis status, and runtime metadata. |
| `GET /v1/cache/stats` | Redis operational counters and memory information. Protect this endpoint before deployment. |
| `POST /v1/query` | JSON query with answer, citations, confidence, and telemetry. |
| `POST /v1/query/stream` | SSE query stream with telemetry, tokens, citations, and completion events. |
| `POST /v1/transcribe` | Transcribes a short browser audio recording into query text. |
| `GET /v1/documents` | Lists files present in `data/raw/` and their indexed chunk counts. |
| `POST /v1/documents/upload` | Saves an uploaded file and runs incremental ingestion. |
| `DELETE /v1/documents/{filename}` | Removes a raw document and its indexed chunks. |
| `GET /v1/models` | Lists supported model identifiers and current process configuration. |
| `POST /v1/models/config` | Updates process-local runtime settings. Protect and validate this endpoint. |
| `POST /v1/memory/clear?session_id=...` | Clears a named Redis-backed session. |

Example query:

```bash
curl -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does this document say about onboarding?","session_id":"demo-session"}'
```

Speech-to-text is configured with `STT_ENABLED`, `STT_BACKEND`, `GROQ_STT_MODEL`,
`STT_MAX_FILE_SIZE_BYTES`, and the optional `STT_DEFAULT_LANGUAGE` environment variables.
The default model is Groq's multilingual `whisper-large-v3-turbo`. The endpoint accepts
`webm`, `wav`, `mp3`, `m4a`, `ogg`, `flac`, and `mp4` recordings and enforces the configured
file-size limit before forwarding audio to the provider. Microphone capture still requires
the user to grant browser permission and the frontend must run on HTTPS or localhost.

Example response fields:

```json
{
  "answer": "...",
  "full_output": "...\n\n📚 Sources & Citations: ...",
  "citations": [],
  "confidence_score": 0.87,
  "telemetry": {
    "latency_ms": 1234.5,
    "retrieved_chunks": 8,
    "reranked_top_n": 3,
    "top_confidence_score": 0.87,
    "llm_backend": "groq",
    "model": "openai/gpt-oss-120b"
  },
  "cached": false,
  "refusal": false
}
```

The current API model includes a `use_cache` request field, but answer-cache reads and writes are not currently wired into the query handler. Treat `cached` as an implementation field rather than a latency guarantee.

The health endpoint checks dependency connectivity and client initialization; it is not a full end-to-end model probe. Runtime model configuration is process-local, and generated requests currently follow the configured `GROQ_MODEL` backend setting.

## Ingestion details

`npm run ingest` compares SHA-256 file hashes with the indexed sources. New files are parsed and embedded, changed files have their old chunks removed before replacement ingestion, and files missing from `data/raw/` have their chunks purged.

The API and interactive CLI use the same hash-based synchronizer automatically. The API starts a background watcher at launch and checks for a change immediately before each document query. Set `AUTO_SYNC_ENABLED=false` to disable this behavior. `AUTO_SYNC_INTERVAL_SECONDS` controls the background polling interval; the minimum supported value is 0.25 seconds.

`npm run watch` performs an initial sync and then polls `data/raw/` every two seconds by default.

Supported source extensions are `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.txt`, `.md`, `.html`, and `.htm`.

Current limitations:

- Scanned or image-only PDFs require an OCR stage that is not included.
- Upload and automatic folder ingestion run synchronously in the indexing process; large files should be moved to a background job system before production use.
- Failed replacement ingestion can leave a modified document absent until a later successful sync.
- Source page metadata is not consistently preserved for every parser, so citations may not identify a PDF page.
- Processed extracted text is stored under `data/processed/`; deletion workflows should remove it as part of customer data deletion.

## Evaluation and testing

Run the tests with:

```bash
npm test
```

The checked-in suite currently contains 69 tests covering normalization, chunking, guardrails, reranking, memory, caching helpers, API behavior, and evaluation helpers. Evaluation tests require the `all-MiniLM-L6-v2` embedding model to be available locally or downloadable at test time.

Run the benchmark evaluator with:

```bash
npm run eval
```

Reports are written to `data/eval_reports/`. The benchmark questions and expected source names must match the documents currently indexed; otherwise a safe refusal can be correct behavior while still producing a low benchmark score.

Run confidence calibration with:

```bash
npm run calibrate
```

Use a representative in-domain and out-of-domain dataset before choosing a threshold. A sigmoid of a cross-encoder score is not automatically a calibrated probability.

## Project structure

```text
RAG-pipeline/
├── api/
│   └── server.py              # FastAPI JSON and SSE service
├── config/
│   └── settings.py            # Environment-backed configuration
├── core/
│   └── cache.py               # Redis chunk, embedding, and session helpers
├── ingestion/
│   ├── document_loader.py     # Unstructured parser and PDF fallback
│   ├── normalizer.py          # Text cleanup and hashes
│   ├── chunker.py             # Hierarchical, section, and recursive chunking
│   ├── embedder.py            # Sentence-transformers embeddings
│   ├── incremental_ingest.py  # Hash-based synchronization
│   └── watcher.py             # Polling file watcher
├── retrieval/
│   ├── vector_store.py        # PostgreSQL schema and hybrid search
│   ├── retriever.py           # Retrieval error handling
│   ├── reranker.py            # ONNX/PyTorch cross-encoder reranking
│   └── health_check.py        # Dependency checks
├── generation/
│   ├── guardrails.py          # Input, context, confidence, and output safety
│   ├── memory.py              # Conversation memory
│   ├── rephraser_agent.py     # Query cleanup and follow-up rewriting
│   ├── query_condenser.py     # Query condensation
│   ├── prompt_builder.py      # Context and prompt construction
│   ├── citation_resolver.py   # Citation extraction and formatting
│   └── llm_client.py          # Groq-compatible LLM client
├── evaluation/
│   ├── evaluator.py           # Local retrieval and answer metrics
│   ├── run_eval.py            # Benchmark runner
│   └── calibrate_threshold.py # Threshold sweep
├── tests/
│   └── test_pipeline.py       # Automated tests
├── data/
│   ├── raw/                   # Source document drop zone
│   ├── processed/             # Extracted normalized text
│   ├── chunks/                # Reserved chunk artifacts directory
│   ├── models/                # Local model artifacts
│   └── eval_reports/          # Evaluation JSON reports
├── docker-compose.yml         # PostgreSQL/Redis development services
├── main.py                    # CLI entrypoint
└── package.json               # npm command wrapper
```

## Production hardening still required

Before serving untrusted or multi-customer traffic, add:

- Authentication, authorization, tenant-aware document storage, tenant-scoped retrieval, and tenant-scoped Redis keys.
- Safe filename handling, extension/content validation, upload size limits, malware scanning, and protected administrative endpoints.
- PostgreSQL connection pooling, request timeouts, retries, rate limits, quotas, cancellation handling, and a background ingestion queue.
- Transactional or staged document replacement with rollback and complete deletion of processed artifacts.
- Structured logs, per-stage traces, audit events, backups, restore drills, schema migrations, and CI lint/type/test checks.
- Claim-level citation validation and an evaluation set tied to the target customer domain.
- Production Docker secrets, private database networking, TLS, non-default credentials, and resource limits.

The repository roadmap in [`TODO.md`](TODO.md) tracks the remaining frontend, mobile, observability, and CI work.

## License

This project is distributed under the GNU Affero General Public License v3.0. See [`LICENSE`](LICENSE). Review the license obligations before offering a closed-source hosted service or distributing modified binaries.
