import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

import redis

from config.settings import (
    REDIS_CACHE_TTL_SECONDS,
    REDIS_ENABLED,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
)

logger = logging.getLogger(__name__)

_redis_client: Optional[redis.Redis] = None
_redis_healthy: Optional[bool] = None
_last_failed_attempt: float = 0.0
_RETRY_COOLDOWN_SECONDS: float = 30.0


def get_redis_client() -> Optional[redis.Redis]:
    """
    Returns a singleton Redis client with lazy connection and quick timeout.
    Implements a circuit-breaker cooldown so offline Redis never blocks queries.
    Returns None if Redis is disabled or unreachable.
    """
    global _redis_client, _redis_healthy, _last_failed_attempt

    if not REDIS_ENABLED:
        return None

    # Circuit breaker: if recent connection failed, skip reconnecting until cooldown passes
    now = time.time()
    if _redis_healthy is False and (now - _last_failed_attempt) < _RETRY_COOLDOWN_SECONDS:
        return None

    if _redis_client is None:
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD if REDIS_PASSWORD else None,
                decode_responses=True,
                socket_connect_timeout=0.25,
                socket_timeout=0.5,
            )
            # Fast ping check
            client.ping()
            _redis_client = client
            _redis_healthy = True
            logger.info(f"Connected to Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.debug(f"Redis not available ({e}). Running without Redis caching.")
            _redis_healthy = False
            _last_failed_attempt = now
            _redis_client = None

    return _redis_client


def is_redis_available() -> bool:
    """
    Checks if Redis cache is online and responding to pings.
    """
    client = get_redis_client()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except Exception:
        global _redis_healthy, _last_failed_attempt, _redis_client
        _redis_healthy = False
        _last_failed_attempt = time.time()
        _redis_client = None
        return False


def _hash_key(prefix: str, content: str) -> str:
    """
    Creates a deterministic SHA-256 cache key.
    """
    digest = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


# =========================================================================
# Tier 1: Document Chunk Cache (Pre-computed Hierarchical/Semantic Chunks)
# =========================================================================

def get_cached_chunks(
    doc_hash: str,
    strategy: str = "hierarchical",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieves cached pre-computed document chunks from Redis by document hash and chunking strategy.
    Returns None on cache miss or when Redis is offline.
    """
    client = get_redis_client()
    if not client or not doc_hash:
        return None

    try:
        sub_key = f"{doc_hash}:{strategy}:{chunk_size or 0}:{chunk_overlap or 0}"
        key = _hash_key("rag:chunks", sub_key)
        raw = client.get(key)
        if raw:
            chunks = json.loads(raw)
            if isinstance(chunks, list):
                return chunks
    except Exception as e:
        logger.debug(f"Tier-1 chunk cache read failed: {e}")

    return None


def set_cached_chunks(
    doc_hash: str,
    chunks: List[Dict[str, Any]],
    strategy: str = "hierarchical",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    ttl: int = REDIS_CACHE_TTL_SECONDS,
):
    """
    Caches pre-computed document chunks and metadata in Redis.
    """
    client = get_redis_client()
    if not client or not doc_hash or not chunks:
        return

    try:
        sub_key = f"{doc_hash}:{strategy}:{chunk_size or 0}:{chunk_overlap or 0}"
        key = _hash_key("rag:chunks", sub_key)
        client.set(key, json.dumps(chunks), ex=ttl)
    except Exception as e:
        logger.debug(f"Tier-1 chunk cache write failed: {e}")


def invalidate_cached_chunks(doc_hash: str):
    """
    Invalidates all cached chunks for a specific document hash in Redis.
    """
    client = get_redis_client()
    if not client or not doc_hash:
        return

    try:
        # Delete direct strategy keys or scan matching keys
        for strategy in ("hierarchical", "semantic_section", "recursive", "structural"):
            for cs in (0, 400, 800, 2000):
                for co in (0, 80, 150, 200):
                    sub_key = f"{doc_hash}:{strategy}:{cs}:{co}"
                    key = _hash_key("rag:chunks", sub_key)
                    client.delete(key)
    except Exception as e:
        logger.debug(f"Tier-1 chunk cache invalidation failed: {e}")


# Compatibility helpers for deprecated answer cache
def get_cached_answer(query: str, scope_hash: str = "global") -> Optional[Dict[str, Any]]:
    return None


def set_cached_answer(
    query: str,
    answer_payload: Dict[str, Any],
    scope_hash: str = "global",
    ttl: int = REDIS_CACHE_TTL_SECONDS,
):
    pass


# =========================================================================
# Tier 2: Embedding Vector Cache
# =========================================================================

def get_cached_embedding(text: str) -> Optional[List[float]]:
    """
    Retrieves cached 384d embedding vector for a snippet of text.
    """
    client = get_redis_client()
    if not client:
        return None

    try:
        key = _hash_key("rag:embed", text)
        raw = client.get(key)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.debug(f"Tier-2 vector cache read failed: {e}")

    return None


def set_cached_embedding(
    text: str,
    vector: List[float],
    ttl: int = REDIS_CACHE_TTL_SECONDS,
):
    """
    Caches 384d embedding vector in Redis.
    """
    client = get_redis_client()
    if not client:
        return

    try:
        key = _hash_key("rag:embed", text)
        client.set(key, json.dumps(vector), ex=ttl)
    except Exception as e:
        logger.debug(f"Tier-2 vector cache write failed: {e}")


# =========================================================================
# Tier 3: Distributed Multi-User Session Memory
# =========================================================================

def get_cached_session(session_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieves multi-turn conversation turns for a session from Redis.
    """
    client = get_redis_client()
    if not client:
        return None

    try:
        key = f"rag:session:{session_id}"
        raw = client.get(key)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.debug(f"Tier-3 session read failed: {e}")

    return None


def set_cached_session(
    session_id: str,
    turns: List[Dict[str, Any]],
    ttl: int = 86400,  # 24 hours
):
    """
    Persists multi-turn conversation turns for a session in Redis.
    """
    client = get_redis_client()
    if not client:
        return

    try:
        key = f"rag:session:{session_id}"
        client.set(key, json.dumps(turns), ex=ttl)
    except Exception as e:
        logger.debug(f"Tier-3 session write failed: {e}")


def clear_cached_session(session_id: str):
    """
    Clears session memory for a given session ID in Redis.
    """
    client = get_redis_client()
    if not client:
        return

    try:
        key = f"rag:session:{session_id}"
        client.delete(key)
    except Exception as e:
        logger.debug(f"Tier-3 session clear failed: {e}")


# =========================================================================
# Cache Diagnostics & Stats
# =========================================================================

def get_cache_stats() -> Dict[str, Any]:
    """
    Returns Redis cache operational statistics for health monitoring.
    """
    client = get_redis_client()
    if not client:
        return {
            "status": "offline",
            "enabled": REDIS_ENABLED,
            "host": f"{REDIS_HOST}:{REDIS_PORT}",
        }

    try:
        info = client.info()
        return {
            "status": "online",
            "enabled": REDIS_ENABLED,
            "host": f"{REDIS_HOST}:{REDIS_PORT}",
            "version": info.get("redis_version", "unknown"),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "connected_clients": info.get("connected_clients", 0),
            "total_commands_processed": info.get("total_commands_processed", 0),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "host": f"{REDIS_HOST}:{REDIS_PORT}",
        }
