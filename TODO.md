# 📋 Enterprise RAG Pipeline — Master TODO & Roadmap

## ✅ Completed Milestones (Production Core)

- [x] **Core RAG Pipeline**: Hierarchical Parent-Child Chunking, HNSW PostgreSQL `pgvector` index, GIN `tsvector` Full-Text Search, and Reciprocal Rank Fusion (RRF).
- [x] **Precision Reranker**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) with Sigmoid confidence calibration.
- [x] **Multi-Layer Guardrails System**:
  - [x] Layer 1 Input Guardrail (Homoglyph, Leetspeak, Base64 payload decoding, Delimiter tags).
  - [x] Layer 2 Conversational Chit-Chat Fast Router (0ms vector overhead).
  - [x] Layer 3 Indirect Prompt Injection Sanitizer for uploaded document passages (`[BLOCKED CONTENT]`).
  - [x] Layer 4 Zero-Hallucination Confidence Gate ($\theta = 0.10$).
  - [x] Layer 5 Pre-Display Output Guardrail (Secret/API key redaction, system prompt leak scrubbing).
- [x] **Conversational Memory**: 5-turn sliding window with follow-up query condensation.
- [x] **Verifiable Citation Resolver**: Inline bracket resolution with transparent quote & confidence footer.
- [x] **Smart Ingestion Engine**: SHA-256 hash diffing (<50ms skip for unchanged docs) + Real-time folder watcher.
- [x] **Cross-Platform Compatibility**: Universal support for Windows (`rag.bat`, `rag.ps1`), macOS/Linux (`./rag`), and `npm run` scripts.
- [x] **Docker Integration**: Single-command container setup via `docker-compose.yml`.
- [x] **Automated Test Suite**: 45/45 Unit Tests passing across all modules (`npm test`).
- [x] **Hugging Face Hub Auth**: Authenticated model downloads via `HF_TOKEN` with clean logs.

---

## 🚀 Phase 1: Performance & Latency Optimization (Target: <200ms TTFT)

- [x] **1.1 Dual-Engine LLM Router**:
  - Add optional Groq API connector (`llama-3.3-70b` / `llama-3.1-8b-instant`) in `llm_client.py` switchable via `.env` (`LLM_BACKEND=ollama|groq`).
  - Keep 100% local air-gapped Ollama as default fallback.
- [x] **1.2 Parallelize Vector + BM25 Hybrid Retrieval**:
  - Concurrently run dense vector search and BM25 lexical search via thread-pool executor in `vector_store.py` (shaves ~40ms).
- [x] **1.3 ONNX Runtime Cross-Encoder Quantization**:
  - Export `ms-marco-MiniLM-L-6-v2` to ONNX format with graph optimization in `reranker.py` for 3–4× faster CPU reranking.
- [x] **1.4 Fast Heuristic Query Condensation**:
  - Fast 0ms standalone query bypass and heuristic context stitching in `query_condenser.py` for follow-ups (saves ~2-3 seconds).

---

## ⚡ Phase 2: Production API & Caching Layer (FastAPI + Redis)

- [ ] **2.1 Redis Multi-Tier Caching**:
  - Add Redis service to `docker-compose.yml`.
  - Implement Tier-1 Answer Cache (SHA-256 query hash $\rightarrow$ ~2ms instant cache hits).
  - Implement Tier-2 Embedding Vector Cache.
  - Implement Distributed Multi-User Session Memory in Redis.
- [ ] **2.2 Asynchronous FastAPI Web Service**:
  - `POST /v1/query`: Standard JSON query endpoint.
  - `POST /v1/query/stream`: Server-Sent Events (SSE) token streaming.
  - `POST /v1/ingest/upload`: Multipart file upload endpoint for drag-and-drop ingestion.
  - `GET /v1/health`: Liveness and readiness probes for Kubernetes/Docker.
  - CORS middleware enabled for decoupled frontend clients.

---

## 🎨 Phase 3: Decoupled Modern Frontend (Separate Repository)

- [ ] **3.1 Initialize `RAG-Frontend` Repository**:
  - Scaffold Next.js 15 / React 19 app with TailwindCSS, Lucide Icons, and dark mode.
- [ ] **3.2 Interactive Chat Interface**:
  - Real-time streaming response bubbles with markdown rendering and syntax highlighting.
  - Interactive citation badges (click citation $\rightarrow$ opens PDF passage preview).
- [ ] **3.3 Document Management Dashboard**:
  - Drag-and-drop file upload zone with instant ingestion progress bars.
  - Document library table with delete, re-index, and chunk inspection capabilities.

---

## 🛡️ Phase 4: Enterprise Observability & Deployment

- [ ] **4.1 Structured Logging & Tracing**:
  - Replace `print()` with structured JSON logging (`structlog`).
  - Integrate OpenTelemetry per-stage latency tracing (Embed $\rightarrow$ Search $\rightarrow$ Rerank $\rightarrow$ LLM).
- [ ] **4.2 Automated GitHub Actions CI/CD Pipeline**:
  - Automated linting (`ruff`), type-checking (`mypy`), and test execution on every pull request.
