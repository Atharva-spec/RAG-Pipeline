import os
import socket
from typing import Tuple

from config.settings import (
    DATABASE_URL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_BACKEND,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_TABLE,
)


def check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def check_postgres() -> Tuple[bool, str]:
    if not check_port(POSTGRES_HOST, POSTGRES_PORT):
        return False, f"Port {POSTGRES_PORT} unreachable. Run: npm run docker:up (or docker start rag-postgres)"
    try:
        from retrieval.vector_store import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                return True, f"Connected to PostgreSQL ({POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB})"
    except Exception as e:
        return False, f"PostgreSQL connection error: {e}"


def check_llm() -> Tuple[bool, str]:
    if not GROQ_API_KEY:
        return False, "GROQ_API_KEY is not set in .env"
    try:
        from generation.llm_client import _get_groq_client

        client = _get_groq_client()
        if client:
            return True, f"Groq Cloud API Online (Model: '{GROQ_MODEL}')"
        return False, "Failed to initialize Groq client"
    except Exception as e:
        return False, f"Groq API error ({e})"


def run_preflight_check() -> bool:
    """
    Runs quick health check on background services and prints a clean status table.
    Returns True if all critical services are healthy, False otherwise.
    """
    pg_ok, pg_msg = check_postgres()
    llm_ok, llm_msg = check_llm()

    all_healthy = pg_ok and llm_ok

    print("\n" + "=" * 60)
    print(" 🔍 System Pre-Flight Health Check")
    print("=" * 60)

    # PostgreSQL status
    if pg_ok:
        print(f"  [✓] PostgreSQL (pgvector Store): ONLINE ({POSTGRES_HOST}:{POSTGRES_PORT})")
    else:
        print(f"  [✗] PostgreSQL (pgvector Store): OFFLINE")
        print(f"      👉 Fix: Run 'npm run docker:up'")

    # LLM status
    if llm_ok:
        print(f"  [✓] LLM Engine (GROQ CLOUD):       ONLINE ({llm_msg})")
    else:
        print(f"  [✗] LLM Engine (GROQ CLOUD):       OFFLINE ({llm_msg})")
        print(f"      👉 Fix: Ensure GROQ_API_KEY is properly set in .env")

    print("=" * 60)

    if not all_healthy:
        print("\n  ⚠️ Warning: One or more background services are offline.")
        print("  Please start them using the commands above for full functionality.\n")

    return all_healthy

