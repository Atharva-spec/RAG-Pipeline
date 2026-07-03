Native RAG Pipeline

A fully local, end-to-end Retrieval-Augmented Generation (RAG) pipeline — no paid APIs, no cloud dependencies. Ask questions about your own documents and get accurate, grounded answers powered by a local LLM running entirely on your machine.


Architecture

Document Workflow

PDF / TXT --> Parse & Clean --> Chunk --> Embed --> OpenSearch Index

Query Workflow

Question
    |
    v
Embed Query (all-MiniLM-L6-v2)
    |
    v
OpenSearch KNN Search --> Top 10 chunks
    |
    v
Cross-Encoder Reranker --> Top 3 chunks
    |
    v
Prompt Builder
    |
    v
Ollama LLaMA 3.2
    |
    v
Answer

Two-stage retrieval is the core design decision:


Stage 1 — KNN Vector Search: OpenSearch finds the top-10 semantically similar chunks fast using exact KNN
Stage 2 — Cross-Encoder Reranking: Scores each (question + chunk) pair precisely and keeps the best 3 for the LLM — giving both speed and accuracy



Tech Stack

LayerToolWhyLLMOllama (LLaMA 3.2)Free, fully local, no API key neededVector DBOpenSearch (Exact KNN)Production-grade, scalable, Docker-basedEmbeddingsall-MiniLM-L6-v2Fast, offline, 384-dim vectorsRerankerms-marco-MiniLM-L-6-v2Cross-encoder, high precisionPDF ParsingpypdfLightweight, no external dependenciesChunkingLangChain RecursiveTextSplitterRespects sentence boundariesFrameworkLangChain + LangChain HuggingFaceEmbedding and text splitting utilities


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


Free Stack — Zero Cost

ComponentModelHow it runsLLMLLaMA 3.2via Ollama — runs on your CPU/GPU locallyEmbeddingsall-MiniLM-L6-v2via sentence-transformers — fully offlineRerankerms-marco-MiniLM-L-6-v2via sentence-transformers — fully offlineVector DBOpenSearchvia Docker — local container on port 9200

No OpenAI key. No Pinecone. No Cohere. Everything runs on your machine.


Configuration

All settings live in config/settings.py — change once and it updates everywhere.

SettingDefaultWhat it controlsCHUNK_SIZE300Characters per chunk. Smaller = more precise retrievalCHUNK_OVERLAP100Context preserved across chunk boundariesTOP_K15Candidates fetched from OpenSearchRERANK_TOP_N5Chunks passed to the LLM after rerankingLLM_MODELllama3.2Swap to mistral, phi3, gemma2 anytimeEMBEDDING_DIM384Must match your embedding model output size

Every Session - Run in This Order

Open three separate terminals:

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

bashpython -c "from retrieval.vector_store import delete_index; delete_index()"

Always wipe the index before re-ingesting to avoid duplicate or stale chunks.

Author

Atharva — GitHub