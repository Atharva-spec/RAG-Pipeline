import math
import os
import logging

# Suppress HuggingFace / Tokenizers noisy logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import CrossEncoder
from config.settings import RERANK_MODEL, RERANK_TOP_N

_model = None  # lazy-loaded once


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = RERANK_TOP_N,
) -> list[dict]:
    if not chunks:
        return []
    model = _get_model()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs, show_progress_bar=False)


    for chunk, score in zip(chunks, scores):
        raw_score = float(score)
        chunk["rerank_score"] = round(raw_score, 4)
        chunk["confidence_score"] = round(_sigmoid(raw_score), 4)

    ranked = sorted(chunks, key=lambda x: x["confidence_score"], reverse=True)
    return ranked[:top_n]


