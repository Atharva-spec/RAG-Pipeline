import os
import socket
import urllib.request
import ollama
from opensearchpy import OpenSearch
from config.settings import OPENSEARCH_HOST, OPENSEARCH_PORT, OLLAMA_BASE_URL, LLM_MODEL


def check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def check_opensearch() -> tuple[bool, str]:
    if not check_port(OPENSEARCH_HOST, OPENSEARCH_PORT):
        return False, f"Port {OPENSEARCH_PORT} unreachable. Run: docker start native-rag-pipeline"
    try:
        client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            use_ssl=False,
            timeout=2,
        )
        if client.ping():
            return True, f"Connected to OpenSearch at {OPENSEARCH_HOST}:{OPENSEARCH_PORT}"
        return False, "OpenSearch did not respond to ping"
    except Exception as e:
        return False, f"OpenSearch connection error: {e}"


def check_ollama() -> tuple[bool, str]:
    try:
        response = ollama.list()
        models = [m.model for m in response.models] if hasattr(response, "models") else []
        model_names = [str(m).split(":")[0] for m in models]
        
        target_base = LLM_MODEL.split(":")[0]
        has_model = any(target_base in name for name in model_names)
        
        if has_model:
            return True, f"Connected to Ollama (Model '{LLM_MODEL}' ready)"
        return True, f"Connected to Ollama (Note: model '{LLM_MODEL}' may need 'ollama pull {LLM_MODEL}')"
    except Exception as e:
        return False, f"Ollama unreachable ({e}). Run: ollama serve"


def run_preflight_check() -> bool:
    """
    Runs quick health check on background services and prints a clean status table.
    Returns True if all critical services are healthy, False otherwise.
    """
    os_ok, os_msg = check_opensearch()
    ollama_ok, ollama_msg = check_ollama()

    all_healthy = os_ok and ollama_ok

    print("\n" + "=" * 60)
    print(" 🔍 System Pre-Flight Health Check")
    print("=" * 60)
    
    # OpenSearch status
    if os_ok:
        print(f"  [✓] OpenSearch (Vector Store): ONLINE ({OPENSEARCH_HOST}:{OPENSEARCH_PORT})")
    else:
        print(f"  [✗] OpenSearch (Vector Store): OFFLINE")
        print(f"      👉 Fix: Run 'docker start native-rag-pipeline'")

    # Ollama status
    if ollama_ok:
        print(f"  [✓] Ollama (LLM Engine):       ONLINE ({ollama_msg})")
    else:
        print(f"  [✗] Ollama (LLM Engine):       OFFLINE")
        print(f"      👉 Fix: Open a terminal and run 'ollama serve'")

    print("=" * 60)

    if not all_healthy:
        print("\n  ⚠️ Warning: One or more background services are offline.")
        print("  Please start them using the commands above for full functionality.\n")

    return all_healthy
