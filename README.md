# Native RAG Pipeline

A fully local, end-to-end Retrieval-Augmented Generation (RAG) pipeline — no paid APIs, no cloud dependencies. Drop in your own PDFs and text files, ingest them into a vector store, and ask questions grounded in your documents using a local LLM.

## How It Works

```
Documents (PDF/txt) → chunk (800 chars) → embed → OpenSearch (HNSW + BM25)
                                                      ↓
Question → embed → Hybrid Search (Vector + BM25 via RRF) → Cross-Encoder rerank → Ollama (llama3.2) → answer
```

## Prerequisites

| Tool | Purpose |
|------|---------|
| Python 3 | Pipeline runtime |
| Docker | OpenSearch vector store |
| Ollama | Local LLM server |

Install Ollama from [ollama.com/download](https://ollama.com/download) or via Homebrew:

```bash
brew install ollama
```

Do **not** run `brew services start ollama` if you plan to use `ollama serve` manually — both bind to port `11434` and will conflict.

## One-Time Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd RAG-pipeline
```

### 2. Python environment

**Option A — npm wrapper**

```bash
npm install
```

Creates `venv/` and installs dependencies from `requirements.txt`. Works on macOS/Linux and Windows as long as Python 3 is on your PATH.

**Option B — direct setup (macOS/Linux)**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Option B — direct setup (Windows)**

```bat
py -3 -m venv venv
venv\Scripts\activate
venv\Scripts\pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
```

### 4. Pull the LLM model

The default model is `llama3.2` (set in `config/settings.py`):

```bash
ollama pull llama3.2
```

### 5. Start OpenSearch (first time only)

If the container does not exist yet, create it:

```bash
docker run -d \
  --name native-rag-pipeline \
  -p 9200:9200 \
  -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_INSTALL_DEMO_CONFIG=true" \
  -e "plugins.security.disabled=true" \
  -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" \
  opensearchproject/opensearch:latest
```

Verify it is running:

```bash
curl http://localhost:9200
```

---

## Every Session

Open **three terminals** and run these in order.

### Terminal 1 — OpenSearch

```bash
docker start native-rag-pipeline
```

Wait until the container is up:

```bash
curl http://localhost:9200
```

### Terminal 2 — Ollama

```bash
ollama serve
```

Leave this terminal open. If you see `bind: address already in use`, a background Ollama process is already running — stop it first:

```bash
brew services stop ollama
```

### Terminal 3 — Project

```bash
cd RAG-pipeline
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows
```

---

## Ingestion

Ingestion loads documents from `data/raw/`, splits them into chunks, embeds them, and stores them in OpenSearch.

### Step 1 — Add your documents

Drop PDF or `.txt` files into `data/raw/`:

```
data/raw/
├── report.pdf
└── notes.txt
```

### Step 2 — Run ingestion
With all three services running (OpenSearch, Ollama, venv activated):

```bash
python main.py --ingest
```

To wipe an existing index and perform a clean re-ingestion from scratch:
```bash
python main.py --rebuild
```

---

## Query

### Interactive Terminal Chat Mode (Recommended)
Run interactive back-and-forth chat without reloading models:
```bash
python main.py --chat
```
*(Or simply `python main.py`)*

### Single-shot Query
Ask a one-off question directly from the terminal:
```bash
python main.py --query "What are the main findings in the report?"
```

---

## Managing Documents

| Task | Steps |
|------|-------|
| Add / Remove / Modify documents | Drop/delete files in `data/raw/` → `python main.py --rebuild` |
| Change chunk or model settings | Edit `config/settings.py` → `python main.py --rebuild` |
| Rebuild index from scratch | `python main.py --rebuild` |

---

## Configuration

All tunables live in `config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `llama3.2` | Ollama model for generation |
| `LLM_TEMPERATURE` | `0.0` | Response randomness |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `HYBRID_SEARCH_ENABLED` | `True` | Combine Dense Vector + BM25 via RRF |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `TOP_K` | `15` | Chunks retrieved from hybrid search |
| `RERANK_TOP_N` | `3` | Chunks sent to the LLM after reranking |
| `HNSW_M` | `24` | HNSW max bidirectional links per node |
| `HNSW_EF_CONSTRUCTION`| `128` | HNSW construction search queue size |
| `HNSW_EF_SEARCH` | `100` | HNSW query search queue size |


After changing chunk or embedding settings, wipe the index and re-ingest.

---

## Project Structure

```
RAG-pipeline/
├── config/
│   └── settings.py              # constants (models, chunk size, paths)
├── data/
│   ├── raw/                     # drop PDFs and txts here
│   ├── processed/               # auto-written: cleaned text per document
│   └── chunks/                  # auto-written: chunked JSON per document
├── ingestion/
│   ├── document_loader.py       # unstructured multi-format parsing (PDF, DOCX, PPTX, TXT, MD, HTML)
│   ├── chunker.py               # recursive text splitting
│   ├── embedder.py              # sentence-transformers embeddings
│   └── ingest_pipeline.py       # load → chunk → embed → store
├── retrieval/
│   ├── vector_store.py          # OpenSearch index and KNN search
│   ├── retriever.py             # query embedding and vector search
│   └── reranker.py              # cross-encoder reranking
├── generation/
│   ├── prompt_builder.py        # system prompt + context + question
│   └── llm_client.py            # Ollama API call
├── main.py                      # CLI (--ingest and --query)
├── .env                         # environment variables (not committed)
├── requirements.txt
└── README.md
```

---

## Troubleshooting

**`ollama: command not found`**
Install Ollama and ensure it is on your PATH: `brew install ollama`

**`bind: address already in use` (port 11434)**
A background Ollama service is already running:
```bash
brew services stop ollama
ollama serve
```

**OpenSearch connection refused**
Start the container: `docker start native-rag-pipeline`

**No documents loaded**
Check that files are in `data/raw/` and are `.pdf` or `.txt`

**Stale or duplicate answers after re-ingesting**
Wipe the index before running `--ingest` again

---

## Author

Atharva — [GitHub](https://github.com/atharva)
