import os
from dotenv import load_dotenv
load_dotenv()
# LLM — Ollama runs 100% locally
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
LLM_MODEL = 'llama3.2'
LLM_TEMPERATURE = 0.0
# Embeddings — sentence-transformers, fully offline
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# Retrieval
TOP_K = 10
RERANK_TOP_N = 3
# Reranker — cross-encoder, fully offline
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
# Paths
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'
DATA_CHUNKS_DIR = 'data/chunks'
CHROMA_DB_DIR = 'data/chroma'
COLLECTION_NAME = 'rag_collection'