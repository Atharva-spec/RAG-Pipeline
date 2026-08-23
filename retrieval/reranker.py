import logging
import math
import os
from typing import Any, List, Optional, Tuple

# Suppress HuggingFace / Tokenizers noisy logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import numpy as np
from config.settings import ONNX_MODEL_DIR, RERANK_MODEL, RERANK_TOP_N, USE_ONNX_RERANKER

logger = logging.getLogger(__name__)

_model: Any = None
_onnx_session: Any = None
_tokenizer: Any = None


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _get_pytorch_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(RERANK_MODEL)
    return _model


def _ensure_onnx_model() -> Tuple[str, Any]:
    """
    Ensures ONNX model exists on disk, exporting from PyTorch if needed.
    Returns (onnx_model_path, tokenizer).
    """
    global _tokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.makedirs(ONNX_MODEL_DIR, exist_ok=True)
    onnx_path = os.path.join(ONNX_MODEL_DIR, "model.onnx")

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)

    if not os.path.exists(onnx_path):
        import torch

        print(f"  [ONNX Exporter] Exporting '{RERANK_MODEL}' to ONNX format...")
        pt_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        pt_model.eval()

        dummy_text = [("Sample query", "Sample document passage")]
        inputs = _tokenizer(dummy_text, padding=True, truncation=True, max_length=512, return_tensors="pt")

        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        }
        dummy_inputs = [inputs["input_ids"], inputs["attention_mask"]]

        if "token_type_ids" in inputs:
            input_names.append("token_type_ids")
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
            dummy_inputs.append(inputs["token_type_ids"])

        torch.onnx.export(
            pt_model,
            tuple(dummy_inputs),
            onnx_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
        )
        print(f"  [ONNX Exporter] Export complete: {onnx_path}")

    return onnx_path, _tokenizer


def _get_onnx_session():
    global _onnx_session, _tokenizer
    if _onnx_session is None:
        import onnxruntime as ort

        onnx_path, tokenizer = _ensure_onnx_model()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        _onnx_session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    return _onnx_session, _tokenizer


def _predict_onnx(pairs: List[Tuple[str, str]]) -> List[float]:
    session, tokenizer = _get_onnx_session()
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    input_names = [inp.name for inp in session.get_inputs()]
    ort_inputs = {k: v for k, v in inputs.items() if k in input_names}
    outputs = session.run(None, ort_inputs)
    logits = outputs[0].squeeze(-1)
    if logits.ndim == 0:
        return [float(logits)]
    return [float(x) for x in logits]


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = RERANK_TOP_N,
) -> list[dict]:
    if not chunks:
        return []

    pairs = [(query, c["text"]) for c in chunks]

    scores: List[float] = []
    if USE_ONNX_RERANKER:
        try:
            scores = _predict_onnx(pairs)
        except Exception as e:
            logger.warning(f"ONNX reranking failed ({e}). Falling back to PyTorch CrossEncoder...")
            pt_model = _get_pytorch_model()
            raw_scores = pt_model.predict(pairs, show_progress_bar=False)
            scores = [float(s) for s in raw_scores]
    else:
        pt_model = _get_pytorch_model()
        raw_scores = pt_model.predict(pairs, show_progress_bar=False)
        scores = [float(s) for s in raw_scores]

    for chunk, score in zip(chunks, scores):
        raw_score = float(score)
        chunk["rerank_score"] = round(raw_score, 4)
        chunk["confidence_score"] = round(_sigmoid(raw_score), 4)

    ranked = sorted(chunks, key=lambda x: x["confidence_score"], reverse=True)
    return ranked[:top_n]
