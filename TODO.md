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
- [x] **Automated Test Suite**: 54/54 Unit Tests passing across all modules (`npm test`).
- [x] **Hugging Face Hub Auth**: Authenticated model downloads via `HF_TOKEN` with clean logs.

---

## 🚀 Phase 1: Performance & Latency Optimization (Target: <200ms TTFT)

- [x] **1.1 Cloud LLM Engine**:
  - Integrated high-speed Cloud LLM (Groq `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`) in `llm_client.py`.
- [x] **1.2 Parallelize Vector + BM25 Hybrid Retrieval**:
  - Concurrently run dense vector search and BM25 lexical search via thread-pool executor in `vector_store.py` (shaves ~40ms).
- [x] **1.3 ONNX Runtime Cross-Encoder Quantization**:
  - Export `ms-marco-MiniLM-L-6-v2` to ONNX format with graph optimization in `reranker.py` for 3–4× faster CPU reranking.
- [x] **1.4 Fast Heuristic Query Condensation**:
  - Fast 0ms standalone query bypass and heuristic context stitching in `query_condenser.py` for follow-ups (saves ~2-3 seconds).

---

## ⚡ Phase 2: Production API & Caching Layer (FastAPI + Redis)

- [x] **2.1 Redis Multi-Tier Caching**:
  - [x] Add Redis container (`redis:7-alpine`) to `docker-compose.yml`.
  - [x] Tier-1 Answer Cache in `core/cache.py` (SHA-256 query hash $\rightarrow$ ~2.7ms instant cache hits).
  - [x] Tier-2 Embedding Vector Cache for repeated semantic searches.
  - [x] Tier-3 Distributed Multi-User Session Memory in Redis with 24h TTL.
  - [x] Graceful zero-exception offline fallback if Redis is unavailable.
- [x] **2.2 Asynchronous FastAPI Web Service**:
  - [x] `POST /v1/query`: Standard JSON query endpoint with telemetry and citation resolution.
  - [x] `POST /v1/query/stream`: Server-Sent Events (SSE) token-by-token streaming endpoint.
  - [x] `POST /v1/documents/upload`: Multipart drag-and-drop file upload with auto-indexing.
  - [x] `GET /v1/documents` & `DELETE /v1/documents/{filename}`: Knowledge Vault file & chunk manager.
  - [x] `GET /v1/models` & `POST /v1/models/config`: Dynamic inference model selection & configuration.
  - [x] `GET /v1/health`: Live readiness probe for PostgreSQL, Groq Cloud, and Redis.
  - [x] `POST /v1/memory/clear`: Session memory reset.
  - [x] CORS middleware pre-configured for Next.js, Stitch, and Expo mobile clients.
  - [x] Single-command execution via `npm run server` / `npm run api`.

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

## 📱 Phase 4: Cross-Platform Mobile App (Expo / React Native)

- [ ] **4.1 Initialize `RAG-Mobile` Repository (Expo SDK 52+ / Expo Router)**:
  - Setup TypeScript, NativeWind (Tailwind CSS for React Native), and Lucide React Native.
- [ ] **4.2 Mobile Conversational Chat & Streaming**:
  - Token streaming interface with haptic feedback on completion.
  - Native bottom-sheet citation inspector (`@gorhom/bottom-sheet`) for inspecting quotes and parent context.
- [ ] **4.3 Mobile Document Management & Camera Ingestion**:
  - Document picker integration (`expo-document-picker`) and file sharing intent handler.
  - Camera document scanner for auto-OCR $\rightarrow$ direct ingestion.
- [ ] **4.4 Offline Caching & Biometric Security**:
  - Secure store for API keys and FaceID / TouchID authentication (`expo-local-authentication`).

---

## 🛡️ Phase 5: Enterprise Observability & Deployment

- [ ] **5.1 Structured Logging & Tracing**:
  - Replace `print()` with structured JSON logging (`structlog`).
  - Integrate OpenTelemetry per-stage latency tracing (Embed $\rightarrow$ Search $\rightarrow$ Rerank $\rightarrow$ LLM).
- [ ] **5.2 Automated GitHub Actions CI/CD Pipeline**:
  - Automated linting (`ruff`), type-checking (`mypy`), and test execution on every pull request.
