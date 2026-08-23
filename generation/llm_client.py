import logging
import os
from typing import Optional

import ollama
from openai import OpenAI

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_BACKEND,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)

_groq_client: Optional[OpenAI] = None


def _get_groq_client() -> Optional[OpenAI]:
    global _groq_client
    if _groq_client is None:
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if api_key:
            _groq_client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
    return _groq_client


def _call_ollama(messages: list[dict], model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE) -> str:
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def _call_groq(messages: list[dict], model: str = GROQ_MODEL, temperature: float = LLM_TEMPERATURE) -> str:
    client = _get_groq_client()
    if not client:
        raise ValueError("GROQ_API_KEY not set or invalid.")
    completion = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        temperature=temperature,
    )
    return completion.choices[0].message.content or ""


def get_answer(
    messages: list[dict],
    backend: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """
    Dispatches generation request to chosen LLM backend (Ollama or Groq).
    If Groq fails or is not configured, gracefully falls back to local Ollama.
    """
    selected_backend = (backend or LLM_BACKEND).lower()

    if selected_backend == "groq":
        try:
            return _call_groq(messages, model=GROQ_MODEL, temperature=temperature)
        except Exception as e:
            logger.warning(f"Groq API call failed ({e}). Falling back to local Ollama...")
            print(f"  [LLM Router Warning] Groq failed: {e}. Falling back to Ollama.")
            return _call_ollama(messages, model=LLM_MODEL, temperature=temperature)

    # Default: Local Ollama
    return _call_ollama(messages, model=LLM_MODEL, temperature=temperature)
