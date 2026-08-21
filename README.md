# Enterprise Agentic Hybrid RAG Pipeline

A fully local, production-grade Conversational Knowledge Engine featuring **HNSW Hybrid Retrieval**, **Hierarchical Multi-Strategy Chunking**, **Multi-Layer Guardrails**, and **Conversational Session Memory** — running 100% locally with no paid cloud APIs.

```
Drop in PDFs, Word docs, PPTs, or Notes ──► Chat grounded in your documents with ZERO hallucinations
```

---

## System Architecture

```
                                 User Message
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      LAYER 1: PRE-FLIGHT & GUARDRAILS                     │
│  ├── 1. Prompt Injection & Jailbreak Check (Blocks overrides & exploits)  │
│  └── 2. Chit-Chat / Intent Fast Router (Instant 1ms response for hi/thanks)│
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (If document query)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                 LAYER 2: CONVERSATIONAL MEMORY & CONDENSER                │
│  ├── Multi-turn Session Manager (Sliding-window buffer: 5 turns)          │
│  └── Agentic Query Condenser (Rewrites "What about the 2nd one?"          │
│      into "What did Robert Katz say about Human skills in Lecture 3?")    │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (Standalone Search Query)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                   LAYER 3: HNSW HYBRID RETRIEVAL CORE                     │
│  ├── Dense Vector Search (High-level semantic matching on child chunks)   │
│  ├── BM25 Lexical Search (Exact keyword, code, and name matching)         │
│  └── Reciprocal Rank Fusion (RRF) (Merges dual rankings into top 15)      │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (Top 15 Candidate Chunks)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                  LAYER 4: CROSS-ENCODER RERANKER & GATE                   │
│  ├── Cross-Attention Reranking (`ms-marco-MiniLM-L-6-v2`)                 │
│  ├── Sigmoid Confidence Scoring (Calibrated 0.0 to 1.0 probability)       │
│  └── Zero-Hallucination Gate (If top score < 0.20 ──► Safe Refusal)       │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (Top Confident Chunks)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                LAYER 5: CONTEXT RESOLUTION & LLM GENERATION               │
│  ├── Parent Context Resolution (Fetches full 2,000c parent paragraphs)    │
│  ├── Parent Deduplication (Removes redundant overlapping parents)         │
│  ├── Ollama LLM Generation (`llama3.2` grounded in document context)      │
│  ├── Output Sanitization (Strips internal system markers)                 │
│  └── Update Session Memory Buffer (Appends turn for next conversation)    │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

* **Multi-Format Ingestion (`unstructured`)**: Supports `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.txt`, `.md`, and `.html`, extracting tables, headings, and lists intact.
* **Hierarchical (Parent-Child) Chunking**: Slices documents into small 400-char child snippets for laser-sharp vector indexing, while linking them to 2,000-char parent context blocks for generation.
* **HNSW Hybrid Retrieval with RRF**: Combines Dense Vector search + BM25 Lexical keyword search using Reciprocal Rank Fusion on OpenSearch HNSW graphs.
* **Cross-Encoder Precision Reranking**: Re-scores candidate passages using deep cross-attention with calibrated sigmoid confidence probabilities ($0.0 \dots 1.0$).
* **Multi-Tier Guardrails**: Blocks prompt injections and system overrides, routes greetings in 1ms, and triggers a safe refusal gate when context is missing.
* **Conversational Memory & Query Condenser**: Maintains a 5-turn sliding-window session buffer and autonomously resolves pronouns and follow-up context.
* **Pre-Flight Health Diagnostics**: Automatically verifies connectivity to OpenSearch and Ollama before starting chat sessions.
* **100% Private & Local**: Runs entirely on your Mac/Linux machine via Docker and Ollama.

---

## Prerequisites

| Tool | Purpose | Windows / macOS / Linux Setup |
| :--- | :--- | :--- |
| **Node.js 18+** | Universal installer & runner | [nodejs.org](https://nodejs.org/) (runs `npm i` and CLI shortcuts) |
| **Python 3.10+** | Pipeline runtime | [python.org](https://www.python.org/) / `brew` / `apt` |
| **Docker** | OpenSearch Vector Store | [docker.com](https://www.docker.com/products/docker-desktop/) (`docker compose`) |
| **Ollama** | Local LLM Engine | [ollama.com](https://ollama.com/download) / `brew install ollama` |

---

## ⚡ 1-Minute Quick Start (Windows, macOS & Linux)

### 1. Clone & Install Dependencies
Run a single command — `npm install` automatically creates your Python virtual environment and installs all cross-platform requirements:
```bash
git clone https://github.com/Atharva-spec/RAG-Pipeline.git
cd RAG-pipeline

npm install
```

### 2. Turn on Docker & Ollama
```bash
# 1. Start OpenSearch container (via Docker Compose)
docker compose up -d

# 2. Pull local LLM & start Ollama
ollama pull llama3.2
ollama serve
```

### 3. Launch the Pipeline!
```bash
npm start
```
*(Or use `./rag --chat` on macOS/Linux or `rag.bat --chat` on Windows)*

---

## ⚙️ Daily Operating Guide

### Step 1: Start Background Services
Ensure OpenSearch and Ollama are running:

```bash
# Start OpenSearch Vector Store
docker compose up -d
# (Or: docker start native-rag-pipeline)

# Start Ollama LLM
ollama serve
```

---

### Step 2: Add Documents & Ingest

1. Drop your `.pdf`, `.docx`, `.pptx`, `.txt`, or `.md` files into `data/raw/`:
   ```
   data/raw/
   ├── Employee_Handbook.pdf
   ├── Architecture_Spec.docx
   └── Notes.md
   ```

2. Ingest and build the HNSW index:
   ```bash
   source venv/bin/activate
   python main.py --rebuild
   ```

---

### Step 3: Start Interactive Chat

Run the interactive conversational terminal chat:

#### Smart Incremental Ingestion:
Scans `data/raw/` with SHA-256 hash diffing. Skips unchanged files in **<50ms**, updates modified files, and purges deleted documents:
```bash
python main.py --ingest
```

#### Live Background File Watcher (Hands-Free):
Monitors `data/raw/` in real time. The second you drag & drop a PDF, Word doc, or presentation, it auto-indexes it in the background:
```bash
python main.py --watch
```

#### Rebuild from Scratch:
```bash
python main.py --rebuild
```

#### Interactive Chat Commands:
* `Type your question`: Asks questions with conversational memory and query condensation.
* `/clear` or `clear`: Resets active session memory.
* `exit` or `quit`: Exits the session.

#### Single-Query Mode:
```bash
python main.py --query "What are the managerial skills from Lecture 3?"
```

#### RAGAS Benchmark Evaluation Suite:
Run quantitative local evaluation across Faithfulness, Relevance, Precision, and Recall:
```bash
python main.py --evaluate
```

#### Confidence Threshold Calibration:
Sweep cross-encoder confidence thresholds to find optimal zero-hallucination settings:
```bash
python main.py --calibrate
```



---

## Configuration

All system parameters live in [`config/settings.py`](file:///Users/atharva/RAG-pipeline/config/settings.py):

| Setting | Default | Description |
| :--- | :--- | :--- |
| `LLM_MODEL` | `llama3.2` | Ollama model for generation |
| `LLM_TEMPERATURE` | `0.0` | Response randomness |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNKING_STRATEGY` | `hierarchical` | Chunking mode (`hierarchical`, `semantic_section`, `recursive`) |
| `PARENT_CHUNK_SIZE` | `2000` | Characters per parent context block |
| `CHILD_CHUNK_SIZE` | `400` | Characters per child search snippet |
| `HYBRID_SEARCH_ENABLED`| `True` | Fuses Dense Vector + BM25 via RRF |
| `RRF_K` | `60` | Reciprocal Rank Fusion ranking constant |
| `TOP_K` | `15` | Chunks retrieved from hybrid search |
| `RERANK_TOP_N` | `3` | Chunks sent to the LLM after reranking |
| `HNSW_M` | `24` | HNSW max bidirectional links per node |
| `HNSW_EF_CONSTRUCTION` | `128` | HNSW graph construction queue size |
| `HNSW_EF_SEARCH` | `100` | HNSW query search queue size |
| `HNSW_ENGINE` | `faiss` | HNSW engine (`faiss` or `lucene`) |
| `GUARDRAILS_ENABLED` | `True` | Enables input safety and confidence gates |
| `GUARDRAIL_CONFIDENCE_THRESHOLD` | `0.20` | Minimum normalized confidence required to answer |

---

## Latency Benchmarks (Measured on Apple Silicon)

| Query Type | Retrieval & Guardrails | LLM Generation | Total Response Time |
| :--- | :--- | :--- | :--- |
| **Chit-Chat (Greetings/Thanks)** | 0 ms (Bypassed) | 0 ms | **1 ms** *(Instant)* |
| **Out-of-Scope (Refusal Gate)** | 75 ms | 0 ms (Bypassed) | **75 ms** |
| **First-Turn Document Question** | 80 – 110 ms | 1.2 – 1.8 s | **1.3 – 1.9 seconds** |
| **Multi-Turn Follow-Up (Turn 2+)** | 350 – 450 ms *(incl. rewrite)* | 1.2 – 1.8 s | **1.6 – 2.3 seconds** |

---

## Automated Test Verification

Run the full automated unit test suite anytime:
```bash
source venv/bin/activate
python -m unittest discover -s tests -v
```

All **22/22 unit tests passing (100% OK)** covering chunking, ingestion, hybrid search, reranking, guardrails, memory, query condensation, and health checks.

---

## Project Structure

```
RAG-pipeline/
├── config/
│   └── settings.py              # Central configuration (Models, HNSW, Chunking, Guardrails)
├── ingestion/
│   ├── document_loader.py       # Unstructured multi-format parser (PDF, DOCX, PPTX, TXT, MD, HTML)
│   ├── chunker.py               # Vast multi-strategy chunker (Hierarchical, Section, Recursive)
│   ├── embedder.py              # Sentence-transformers embedding model wrapper
│   └── ingest_pipeline.py       # Ingestion orchestrator
├── retrieval/
│   ├── health_check.py          # Pre-flight diagnostic health checker (OpenSearch & Ollama)
│   ├── vector_store.py          # OpenSearch HNSW ANN index, bulk indexing, Hybrid Search with RRF
│   ├── retriever.py             # Hybrid search query dispatcher and error handling
│   └── reranker.py              # Cross-Encoder reranker with sigmoid confidence scoring
├── generation/
│   ├── guardrails.py            # Input safety, prompt injection detection, intent routing, refusal gate
│   ├── memory.py                # Session conversation memory manager with sliding window buffer
│   ├── query_condenser.py       # Agentic rewriter that transforms follow-up chat into standalone queries
│   ├── prompt_builder.py        # Context builder with parent deduplication and dialog history
│   └── llm_client.py            # Local Ollama API client
├── tests/
│   └── test_pipeline.py         # Automated test suite (22 unit tests covering all components)
├── data/
│   ├── raw/                     # User drop-zone for source documents
│   ├── processed/               # Cleaned extracted text per document
│   └── chunks/                  # Chunked JSON payloads with parent-child metadata
├── main.py                      # Interactive CLI, continuous chat loop, index rebuild options
├── README.md                    # This file
├── HANDOFF_WALKTHROUGH.md       # Comprehensive architectural handoff guide
└── requirements.txt             # Locked Python dependencies
```

---

## Troubleshooting

**`Ollama: command not found`**
* Install Ollama and ensure it is on your PATH: `brew install ollama`

**`Pre-Flight Check: Ollama is OFFLINE`**
* Start the background Ollama server: `ollama serve`

**`Pre-Flight Check: OpenSearch is OFFLINE`**
