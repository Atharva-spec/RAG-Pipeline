Native RAG Pipeline

A fully local, end-to-end Retrieval-Augmented Generation (RAG) pipeline — no paid APIs, no cloud dependencies. Ask questions about your own documents and get accurate, grounded answers powered by a local LLM running entirely on your machine.

Project Structure

rag-pipeline/
│
├── config/
│   ├── __init__.py
│   └── settings.py              ← all constants (chunk size, model names, paths)
│
├── data/
│   ├── raw/                     ← drop your PDFs and txts here manually
│   ├── processed/               ← auto-written: cleaned text per document
│   └── chunks/                  ← auto-written: chunked JSON per document
│
├── ingestion/
│   ├── __init__.py
│   ├── document_loader.py       ← PDF and txt parsing and cleaning
│   ├── chunker.py               ← recursive text splitting into chunks
│   ├── embedder.py              ← sentence-transformers embedding model
│   └── ingest_pipeline.py       ← orchestrates load → chunk → embed → store
│
├── retrieval/
│   ├── __init__.py
│   ├── vector_store.py          ← OpenSearch index creation and KNN search
│   ├── retriever.py             ← query embedding and vector search
│   └── reranker.py              ← cross-encoder reranking
│
├── generation/
│   ├── __init__.py
│   ├── prompt_builder.py        ← assembles system prompt + context + question
│   └── llm_client.py            ← Ollama API call and response handling
│
├── main.py                      ← CLI entry point (--ingest and --query)
├── .env                         ← environment variables (never committed)
├── .env.example                 ← safe template to share
├── .gitignore
├── requirements.txt
└── README.md

Every Session - Run in This Order

Open three separate terminals:

Initial setup

Option 1 — npm wrapper

```bash
npm install
```

This creates `venv/` and installs Python dependencies from `requirements.txt`.
It now supports Windows and macOS/Linux, as long as Python 3 is installed and available in PATH.

Option 2 — direct Python setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows direct setup:

```bat
py -3 -m venv venv
venv\Scripts\activate
venv\Scripts\pip install -r requirements.txt
```

Terminal 1 - Start OpenSearch

bashdocker start native-rag-pipeline

Terminal 2 - Start Ollama

bashollama serve

Terminal 3 - Your project

bashcd rag-pipeline
source venv/bin/activate    # Mac/Linux

Managing Documents

TaskCommandAdd a new PDFDrop into data/raw/ then run --ingest
Remove a documentDelete file from data/raw/ then run command below
Wipe and rebuild indexRun command below then --ingest
Change chunk settings Wipe index then --ingest

Wipe the OpenSearch index:

bash python -c "from retrieval.vector_store import delete_index; delete_index()"

Always wipe the index before re-ingesting to avoid duplicate or stale chunks.

Author

Atharva — GitHub
