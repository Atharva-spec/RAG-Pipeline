import logging
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

# Global HuggingFace Hub / Tokenizers quiet configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", module="huggingface_hub")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# LLM Backend & Cloud Generation
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_MODEL = os.getenv("LLM_MODEL", GROQ_MODEL)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Query Condensation Strategy
FAST_QUERY_CONDENSER = os.getenv("FAST_QUERY_CONDENSER", "true").lower() in ("true", "1", "yes")


# Embeddings & HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

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
USE_ONNX_RERANKER = os.getenv("USE_ONNX_RERANKER", "true").lower() in ("true", "1", "yes")
ONNX_MODEL_DIR = os.getenv("ONNX_MODEL_DIR", "data/models/reranker_onnx")
HYBRID_SEARCH_ENABLED = True
RRF_K = 60  # Reciprocal Rank Fusion constant

# PostgreSQL & pgvector Database Core
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_TABLE = os.getenv("POSTGRES_TABLE", "document_chunks")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)
HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "64"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "40"))



# Guardrails & Safety
GUARDRAILS_ENABLED = True
GUARDRAIL_CONFIDENCE_THRESHOLD = 0.20  # Minimum normalized confidence score (0.0 to 1.0)
GUARDRAIL_BLOCK_PROMPT_INJECTION = True

# Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_CHUNKS_DIR = "data/chunks"


