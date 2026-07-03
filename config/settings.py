import os

from dotenv import load_dotenv

load_dotenv()

# LLM
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = "llama3.2"
LLM_TEMPERATURE = 0.0

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # must match your embedding model output size

# Chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 10
RERANK_TOP_N = 3
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# OpenSearch
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX = "rag_index"

# Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_CHUNKS_DIR = "data/chunks"
