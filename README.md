# 🚀 Enterprise Agentic Hybrid RAG Pipeline

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Database: PostgreSQL pgvector](https://img.shields.io/badge/Database-PostgreSQL_pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![LLM: Groq Cloud](https://img.shields.io/badge/LLM-Groq_Cloud-f55036.svg)](https://groq.com/)
[![Tests: 54 Passing](https://img.shields.io/badge/tests-54%20passed-success.svg)]()

A high-performance, production-ready Conversational Knowledge Engine featuring **PostgreSQL HNSW `pgvector`**, **GIN Full-Text Lexical Search**, **Hierarchical Parent-Child Chunking**, **ONNX Runtime Quantized Reranking**, **Multi-Layer Guardrails**, and **High-Speed Cloud LLM (Groq)**.

```
Drop in PDFs, Word docs, PPTs, or Notes ──► Chat grounded in your documents with ZERO hallucinations
```

---

## 🏗️ System Architecture

```
                                 User Message
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      LAYER 1: PRE-FLIGHT & GUARDRAILS                     │
│  ├── 1. Prompt Injection & Jailbreak Check (Blocks overrides & exploits)  │
│  └── 2. Chit-Chat / Intent Fast Router (Instant 0ms response for hi/thanks)│
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (If document query)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                 LAYER 2: CONVERSATIONAL MEMORY & CONDENSER                │
│  ├── Multi-turn Session Manager (Sliding-window buffer: 5 turns)          │
│  └── Fast Query Condenser (0ms heuristic bypass / Groq context rewriter)  │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (Standalone Search Query)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│             LAYER 3: POSTGRESQL HYBRID RETRIEVAL CORE (`pgvector`)        │
│  ├── Dense Vector Search (HNSW Cosine similarity on 384d child embeddings)│
│  ├── Full-Text Lexical Search (PostgreSQL GIN tsvector keyword matching)  │
│  └── Reciprocal Rank Fusion (RRF) (Parallel execution, merges dual ranks) │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (Top 15 Candidate Chunks)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│               LAYER 4: ONNX CROSS-ENCODER RERANKER & GATE                 │
│  ├── Quantized ONNX Inference (`ms-marco-MiniLM-L-6-v2` 3–4× CPU boost)   │
│  ├── Sigmoid Confidence Scoring (Calibrated 0.0 to 1.0 probability)       │
│  └── Zero-Hallucination Gate (If top score < 0.20 ──► Safe Refusal)       │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (Top Confident Chunks)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                LAYER 5: CONTEXT RESOLUTION & LLM GENERATION               │
│  ├── Parent Context Resolution (Fetches full 2,000c parent paragraphs)    │
│  ├── Parent Deduplication (Removes redundant overlapping parents)         │
│  ├── Cloud LLM Engine (Groq LPU high-speed inference)                     │
│  ├── Output Guardrail & Redaction (Scrubs keys & system prompt leakage)   │
│  └── Verifiable Inline Citations (Direct transparent quote verification)  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

* **Lightweight PostgreSQL + `pgvector` Core**: Uses `pgvector/pgvector:pg16` in Docker (<80 MB RAM), replacing heavy legacy engines.
* **HNSW Hybrid Retrieval with RRF**: Combines dense vector cosine distance + GIN full-text search using Reciprocal Rank Fusion (RRF).
* **High-Speed Cloud LLM Engine**: Ultra-fast inference via **Groq Cloud** (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`).
* **ONNX Runtime Reranker Quantization**: Accelerated Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) with 3–4× faster CPU latency.
* **Hierarchical (Parent-Child) Chunking**: Slices documents into 400-char child snippets for vector indexing while linking 2,000-char parent context blocks for generation.
* **Multi-Layer Guardrails System**: Blocks prompt injections, leetspeak, homoglyphs, and base64 payloads, routes chit-chat in 0ms, and enforces confidence gates.
* **Verifiable Citation Resolver**: Inline numerical citations with transparent confidence scores and quote validation.
* **Smart Incremental Ingestion**: SHA-256 hash diffing skips unchanged files in <5ms with real-time directory watching.

---

## ⚡ 1-Minute Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Atharva-spec/RAG-Pipeline.git
cd RAG-pipeline

npm install
```

### 2. Configure Environment ([`.env`](file:///.env))
Create or edit your `.env` file:
```dotenv
# PostgreSQL + pgvector Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# LLM Backend: "groq"
LLM_BACKEND=groq
GROQ_API_KEY=gsk_your_groq_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

### 3. Start PostgreSQL & Ingest Documents
```bash
# 1. Start lightweight PostgreSQL container (via Docker Compose)
npm run docker:up

# 2. Ingest your documents (drop .pdf, .docx, .txt into data/raw/)
npm run ingest

# 3. Launch interactive terminal chat!
npm start
```

---

## ⚙️ Daily CLI Commands

| Command | Shortcut | Description |
| :--- | :--- | :--- |
| `npm start` | `./rag --chat` | Start interactive conversational chat loop |
| `npm run query -- "..."` | `./rag --query "..."` | Ask a one-off question from terminal |
| `npm run ingest` | `./rag --ingest` | Run smart incremental document ingestion (<5ms skip) |
| `npm run rebuild` | `./rag --rebuild` | Wipe database and re-ingest all documents from scratch |
| `npm run watch` | `./rag --watch` | Start live background folder watcher on `data/raw/` |
| `npm test` | `npm test` | Run full automated test suite (54/54 tests) |
| `npm run docker:up` | `docker compose up -d` | Start PostgreSQL (`pgvector`) container |
| `npm run docker:down` | `docker compose down` | Stop PostgreSQL container |

---

## 🧪 Automated Test Verification

Run the full automated unit test suite anytime:
```bash
npm test
```

```
Ran 54 tests in 7.787s

OK (54 tests passed, 0 failures, 0 errors)
```

---

## 📁 Project Structure

```
RAG-pipeline/
├── config/
│   └── settings.py              # Central configuration (Postgres, Models, Chunking, Guardrails)
├── ingestion/
│   ├── document_loader.py       # Multi-format parser (PDF, DOCX, PPTX, TXT, MD)
│   ├── chunker.py               # Hierarchical, Section, and Recursive chunking strategies
│   ├── embedder.py              # Sentence-transformers embedding wrapper
│   ├── incremental_ingest.py    # Relational hash-diffing synchronization
│   └── watcher.py               # Live directory file watcher
├── retrieval/
│   ├── health_check.py          # Pre-flight diagnostic health checker (PostgreSQL & Groq)
│   ├── vector_store.py          # PostgreSQL pgvector HNSW ANN + GIN tsvector hybrid search
│   ├── retriever.py             # Query dispatcher & error handler
│   └── reranker.py              # ONNX Runtime accelerated Cross-Encoder reranker
├── generation/
│   ├── guardrails.py            # 5-layer safety & prompt injection defense
│   ├── memory.py                # Session conversation memory manager
│   ├── rephraser_agent.py       # Rephraser Agent for typos, noise stripping & context resolution
│   ├── query_condenser.py       # Fast heuristic & LLM query condensation wrapper
│   ├── prompt_builder.py        # Context builder with parent deduplication
│   ├── citation_resolver.py     # Inline quote & transparent confidence resolver
│   └── llm_client.py            # High-Speed Cloud LLM Client (Groq)
├── tests/
│   └── test_pipeline.py         # 54 unit tests covering all pipeline modules
├── data/
│   ├── raw/                     # User drop-zone for source documents
│   ├── processed/               # Cleaned extracted text per document
│   └── models/                  # Exported ONNX model cache
├── docker-compose.yml           # PostgreSQL pgvector container configuration
├── LICENSE                      # GNU Affero General Public License v3.0 (AGPL-3.0)
├── main.py                      # Interactive CLI entrypoint
└── package.json                 # Cross-platform runner scripts
```

---

## ⚖️ License

Distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [`LICENSE`](file:///Users/atharva/RAG-pipeline/LICENSE) for details.
