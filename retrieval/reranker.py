import logging
import math
import os
from typing import Any, List, Tuple

# Suppress HuggingFace / Tokenizers noisy logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

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


def extract_core_entities(query: str) -> List[str]:
    """
    Extracts core candidate entities/concepts from inquiries (e.g., 'what is neem' -> ['neem'],
    'tell me about neem' -> ['neem'], 'What remedy is made from Nimba?' -> ['Nimba']).
    """
    import re
    candidates = []
    q = query.strip().rstrip("?. ")

    # 1. Direct inquiry stripping (e.g. 'what is neem' -> 'neem')
    clean = re.sub(
        r"^(?:what (?:is|are|was|were)(?: the| an| a)?|who (?:is|are)|define|meaning of|information on|details on|tell me about|explain to me|describe)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()
    if clean and clean.lower() != q.lower() and len(clean.split()) <= 4:
        if clean.lower() not in {"this", "that", "these", "those", "it", "something", "anything"}:
            candidates.append(clean)

    # 2. Extract prepositional objects: 'from X', 'using X', 'with X', 'for X', 'about X', 'of X'
    m_prep = re.findall(r"\b(?:from|using|with|about|regarding|of)\s+([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+){0,2})\b", q, flags=re.IGNORECASE)
    stopwords = {"the", "a", "an", "this", "that", "these", "those", "it", "its", "them", "me", "us", "you", "which", "what", "how", "when", "where", "some", "any"}
    for p in m_prep:
        p_clean = p.strip()
        if p_clean.lower() not in stopwords and len(p_clean) > 2 and len(p_clean.split()) <= 3:
            candidates.append(p_clean)

    # 3. Capitalized terms (excluding the first word of sentence)
    words = q.split()
    if len(words) > 1:
        for w in words[1:]:
            clean_w = re.sub(r"[^a-zA-Z0-9_\-]", "", w)
            if clean_w and clean_w[0].isupper() and clean_w.lower() not in stopwords and clean_w.lower() not in {"ayurvedic", "ayurveda"}:
                candidates.append(clean_w)

    return list(dict.fromkeys(candidates))[:2]


def normalize_query_for_reranker(query: str) -> str:
    """
    Normalizes colloquial idioms (e.g. 'full form') and conversational frames so cross-encoders
    (trained on standard MS-MARCO corpora) do not artificially penalize conversational queries.
    """
    import re
    q = query.strip()
    # Strip trailing conversational phrases
    q = re.sub(r"[\s,]+(?:to me|for me|please|can you|could you|if possible)\s*\??$", "", q, flags=re.IGNORECASE).strip()
    # Strip leading conversational frames
    q = re.sub(
        r"^(?:can you (?:please )?(?:tell|explain|describe|give|share|provide)(?: to)?(?: me)?(?: more)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?|could you (?:please )?(?:tell|explain|describe|give|share|provide)(?: to)?(?: me)?(?: more)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?|please (?:tell|explain|describe|give|share|provide)(?: to)?(?: me)?(?: more)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?|tell me(?: more)?(?: about| regarding| on)?|talk (?:to me )?about|explain to me(?: about)?|describe to me(?: about)?|what can you tell me(?: about| regarding)?|what do you know about|what about|how about|i (?:just )?(?:want|would like|wish) to know(?: more)?(?: about| on| regarding)?|give me(?: some)?(?: details| info(?:rmation)?)?(?: about| on| regarding)?|information (?:about|on|regarding)|details (?:about|on|regarding)|brief (?:me on|about)|help me understand(?: about)?)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()
    # "full form of <TERM>"
    q = re.sub(
        r"(?i)\b(?:what is (?:the )?)?full[-\s]*form of\s+([a-zA-Z0-9_\-]+)\b",
        r"what does \1 stand for (\1 acronym meaning)",
        q,
    )
    # "<TERM> full form"
    q = re.sub(
        r"(?i)\b([a-zA-Z0-9_\-]+)\s+full[-\s]*form\b",
        r"what does \1 stand for (\1 acronym meaning)",
        q,
    )
    # Generic "full form"
    q = re.sub(
        r"(?i)\bfull[-\s]*form\b",
        r"acronym meaning definition",
        q,
    )
    return q


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = RERANK_TOP_N,
) -> list[dict]:
    if not chunks:
        return []

    scoring_query = normalize_query_for_reranker(query)
    pairs = [(scoring_query, c["text"]) for c in chunks]

    def _score_pairs(candidate_pairs: List[Tuple[str, str]]) -> List[float]:
        if USE_ONNX_RERANKER:
            try:
                return _predict_onnx(candidate_pairs)
            except Exception as e:
                logger.warning(f"ONNX reranking failed ({e}). Falling back to PyTorch CrossEncoder...")
                pt_model = _get_pytorch_model()
                return [float(s) for s in pt_model.predict(candidate_pairs, show_progress_bar=False)]
        else:
            pt_model = _get_pytorch_model()
            return [float(s) for s in pt_model.predict(candidate_pairs, show_progress_bar=False)]

    scores = _score_pairs(pairs)

    # Dual-scoring for entity inquiries (e.g. 'what is neem', 'What remedy is made from Nimba?'):
    core_entities = extract_core_entities(query)
    for core_ent in core_entities:
        if core_ent and core_ent.lower() != scoring_query.lower():
            pairs_ent = [(core_ent, c["text"]) for c in chunks]
            scores_ent = _score_pairs(pairs_ent)
            scores = [max(s1, s2) for s1, s2 in zip(scores, scores_ent)]

    for chunk, score in zip(chunks, scores):
        raw_score = float(score)
        chunk["rerank_score"] = round(raw_score, 4)
        chunk["confidence_score"] = round(_sigmoid(raw_score), 4)

    ranked = sorted(chunks, key=lambda x: x["confidence_score"], reverse=True)
    return ranked[:top_n]
