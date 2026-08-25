import hashlib
import json
import logging
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


def get_redis_client() -> Optional[redis.Redis]:
    """
    Returns a singleton Redis client with lazy connection and quick timeout.
    Returns None if Redis is disabled or unreachable.
    """
    global _redis_client, _redis_healthy

    if not REDIS_ENABLED:
        return None

    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD if REDIS_PASSWORD else None,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=2.0,
            )
            # Fast ping check
            _redis_client.ping()
            _redis_healthy = True
            logger.info(f"Connected to Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.debug(f"Redis not available ({e}). Running without Redis caching.")
            _redis_healthy = False
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
        return False


def _hash_key(prefix: str, content: str) -> str:
    """
    Creates a deterministic SHA-256 cache key.
    """
    digest = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


# =========================================================================
# Tier 1: Answer & Query Response Cache
# =========================================================================

def get_cached_answer(query: str, scope_hash: str = "global") -> Optional[Dict[str, Any]]:
    """
    Retrieves cached RAG response for a normalized query string.
    Returns None on cache miss or Redis offline.
    """
    client = get_redis_client()
    if not client:
        return None

    try:
        key = _hash_key("rag:answer", f"{scope_hash}:{query}")
        raw = client.get(key)
        if raw:
            data = json.loads(raw)
            data["cached"] = True
            return data
    except Exception as e:
        logger.debug(f"Tier-1 cache read failed: {e}")

    return None


def set_cached_answer(
    query: str,
    answer_payload: Dict[str, Any],
    scope_hash: str = "global",
    ttl: int = REDIS_CACHE_TTL_SECONDS,
):
    """
    Caches RAG query answer and citation payload in Redis.
    """
    client = get_redis_client()
    if not client:
        return

    try:
        key = _hash_key("rag:answer", f"{scope_hash}:{query}")
        payload = dict(answer_payload)
        payload.pop("cached", None)
        client.set(key, json.dumps(payload), ex=ttl)
    except Exception as e:
        logger.debug(f"Tier-1 cache write failed: {e}")


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
