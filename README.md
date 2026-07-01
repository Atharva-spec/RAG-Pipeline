# RAG-Pipeline
🔍 Native RAG Pipeline

A fully local, end-to-end Retrieval-Augmented Generation (RAG) pipeline built from scratch — no paid APIs, no cloud dependencies. Everything runs on your machine.

Ask questions about your own documents and get accurate, grounded answers powered by a local LLM.

🧠 How It Works

DOCUMENT WORKFLOW
─────────────────
PDF / TXT  →  Parse & Clean  →  Chunk  →  Embed  →  OpenSearch Index

QUERY WORKFLOW
──────────────
Question  →  Embed Query  →  KNN Search (top-10)  →  Rerank (top-3)  →  Prompt  →  LLaMA 3.2  →  Answer

Two-stage retrieval is the key design choice:

Stage 1 — Vector Search: OpenSearch KNN finds the top-10 semantically similar chunks fast
Stage 2 — Reranking: A cross-encoder scores each (question, chunk) pair precisely and keeps the best 3 for the LLM

🛠 Tech Stack

LayerToolWhyLLMOllama (LLaMA 3.2)Free, fully local, no API keyVector DBOpenSearch (KNN + HNSW)Production-grade, scalableEmbeddingsall-MiniLM-L6-v2Fast, offline, 384-dimRerankerms-marco-MiniLM-L-6-v2Cross-encoder, high precisionPDF ParsingpypdfLightweight, no dependenciesChunkingLangChain RecursiveTextSplitterRespects sentence boundaries


📁 Project Structure

rag-pipeline/
├── config/
│   └── settings.py          ← all constants (chunk size, model names, paths)
├── data/
│   ├── raw/                 ← drop your PDFs/txts here
│   ├── processed/           ← auto-written: cleaned text per document
│   └── chunks/              ← auto-written: chunked JSON per document
├── ingestion/
│   ├── document_loader.py   ← PDF/txt parsing and cleaning
│   ├── chunker.py           ← recursive text splitting
│   ├── embedder.py          ← sentence-transformers embedding model
│   └── ingest_pipeline.py   ← orchestrates load → chunk → embed → store
├── retrieval/
│   ├── vector_store.py      ← OpenSearch index creation and KNN search
│   ├── retriever.py         ← query embedding + vector search
│   └── reranker.py          ← cross-encoder reranking
├── generation/
│   ├── prompt_builder.py    ← assembles system prompt + context + question
│   └── llm_client.py        ← Ollama API call
├── main.py                  ← CLI entry point
├── .env                     ← OpenSearch credentials (never committed)
└── requirements.txt