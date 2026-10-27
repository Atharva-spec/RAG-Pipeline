import logging
import os
from pathlib import Path
import warnings

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()

# Global HuggingFace Hub / Tokenizers quiet configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
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

# Rephraser Agent & Query Transformation Strategy
REPHRASER_AGENT_ENABLED = os.getenv("REPHRASER_AGENT_ENABLED", "true").lower() in ("true", "1", "yes")
FAST_REPHRASER = os.getenv("FAST_REPHRASER", "true").lower() in ("true", "1", "yes")
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


# Retrieval & Search Core (Sub-200ms Tuned)
TOP_K = 8
RERANK_TOP_N = 3
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_ONNX_RERANKER = os.getenv("USE_ONNX_RERANKER", "true").lower() in ("true", "1", "yes")
ONNX_MODEL_DIR = os.getenv("ONNX_MODEL_DIR", str(BASE_DIR / "data" / "models" / "reranker_onnx"))
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

# Redis Cache Engine
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() in ("true", "1", "yes")
REDIS_CACHE_TTL_SECONDS = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "86400"))  # 24 hour TTL

# API Web Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Absolute Paths anchored to project root
DATA_RAW_DIR = os.getenv("DATA_RAW_DIR", str(BASE_DIR / "data" / "raw"))
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", str(BASE_DIR / "data" / "processed"))
DATA_CHUNKS_DIR = os.getenv("DATA_CHUNKS_DIR", str(BASE_DIR / "data" / "chunks"))


