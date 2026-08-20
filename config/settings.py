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

# Chunking Engine (Vast & Multi-Strategy)
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "hierarchical")  # "hierarchical", "semantic_section", "recursive"
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 80
CHUNK_SIZE = 800  # Fallback for simple recursive splitter
CHUNK_OVERLAP = 150


# Retrieval & Search Core
TOP_K = 15
RERANK_TOP_N = 3
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_SEARCH_ENABLED = True
RRF_K = 60  # Reciprocal Rank Fusion constant

# OpenSearch & HNSW Indexing
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX = "rag_index"
HNSW_M = 24
HNSW_EF_CONSTRUCTION = 128
HNSW_EF_SEARCH = 100
HNSW_ENGINE = "faiss"


# Guardrails & Safety
GUARDRAILS_ENABLED = True
GUARDRAIL_CONFIDENCE_THRESHOLD = 0.20  # Minimum normalized confidence score (0.0 to 1.0)
GUARDRAIL_BLOCK_PROMPT_INJECTION = True

# Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_CHUNKS_DIR = "data/chunks"


