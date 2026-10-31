import logging
import os
from typing import Any, Optional

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_STT_MODEL,
    LLM_BACKEND,
    LLM_MODEL,
    LLM_TEMPERATURE,
    STT_BACKEND,
)

logger = logging.getLogger(__name__)

_groq_client: Optional[Any] = None


def _get_groq_client() -> Optional[Any]:
    global _groq_client
    if _groq_client is None:
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if api_key:
            from openai import OpenAI
            _groq_client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
    return _groq_client


def _call_groq(
    messages: list[dict],
    model: str = GROQ_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    client = _get_groq_client()
    if not client:
        raise ValueError(
            "GROQ_API_KEY is not set or invalid. Please configure GROQ_API_KEY in your .env file."
        )
    completion = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        temperature=temperature,
    )
    return completion.choices[0].message.content or ""


def _stream_groq(
    messages: list[dict],
    model: str = GROQ_MODEL,
    temperature: float = LLM_TEMPERATURE,
):
    client = _get_groq_client()
    if not client:
        raise ValueError(
            "GROQ_API_KEY is not set or invalid. Please configure GROQ_API_KEY in your .env file."
        )
    stream = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
) -> dict[str, Any]:
    """Transcribe a short user recording through the configured STT backend."""
    if STT_BACKEND != "groq":
        raise ValueError(f"Unsupported speech-to-text backend: {STT_BACKEND}")

    client = _get_groq_client()
    if not client:
        raise ValueError(
            "GROQ_API_KEY is not set or invalid. Please configure GROQ_API_KEY in your .env file."
        )

    kwargs: dict[str, Any] = {
        "file": (filename, audio_bytes, content_type or "application/octet-stream"),
        "model": GROQ_STT_MODEL,
        "response_format": "json",
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language

    result = client.audio.transcriptions.create(**kwargs)
    return {
        "text": (getattr(result, "text", "") or "").strip(),
        "language": getattr(result, "language", None),
        "model": GROQ_STT_MODEL,
    }


def get_answer(
    messages: Any,
    backend: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """
    Dispatches generation request to Cloud LLM backend (Groq).
    Accepts either a list of message dicts or a raw string prompt.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    selected_backend = (backend or LLM_BACKEND).lower()

    try:
        model = GROQ_MODEL if selected_backend == "groq" else (LLM_MODEL or GROQ_MODEL)
        return _call_groq(messages, model=model, temperature=temperature)
    except Exception as e:
        logger.error(f"Cloud LLM call failed: {e}")
        raise RuntimeError(f"Cloud LLM generation failed: {e}") from e


def stream_answer(
    messages: Any,
    backend: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
):
    """
    Yields streaming tokens in real time from Cloud LLM backend (Groq).
    Accepts either a list of message dicts or a raw string prompt.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    selected_backend = (backend or LLM_BACKEND).lower()

    try:
        model = GROQ_MODEL if selected_backend == "groq" else (LLM_MODEL or GROQ_MODEL)
        yield from _stream_groq(messages, model=model, temperature=temperature)
    except Exception as e:
        logger.error(f"Cloud LLM streaming failed: {e}")
        raise RuntimeError(f"Cloud LLM streaming failed: {e}") from e
