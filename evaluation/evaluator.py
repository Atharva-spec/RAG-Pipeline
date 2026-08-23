import datetime
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from generation.guardrails import (
    check_input_safety,
    detect_conversational_intent,
    safe_refusal_response,
    sanitize_output,
    validate_output_safety,
    validate_retrieval_confidence,
)
from generation.llm_client import get_answer
from generation.prompt_builder import build_prompt
from generation.citation_resolver import resolve_citations
from ingestion.embedder import get_embedding_model
from retrieval.reranker import rerank_chunks
from retrieval.retriever import retrieve_chunks
from config.settings import RERANK_TOP_N, TOP_K

_eval_embedder = None


def _get_embedder():
    global _eval_embedder
    if _eval_embedder is None:
        _eval_embedder = get_embedding_model()
    return _eval_embedder


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_faithfulness(answer: str, context_chunks: List[str]) -> float:
    """
    Measures Faithfulness (0.0 to 1.0):
    Evaluates what proportion of claims in the generated answer are grounded in the retrieved context.
    """
    if not answer or not context_chunks:
        return 1.0 if not context_chunks and ("don't have enough information" in answer.lower() or "could not find" in answer.lower()) else 0.0

    # Clean citations and footer from answer
    clean_ans = re.sub(r'📚 Sources & Citations:.*', '', answer, flags=re.DOTALL)
    clean_ans = re.sub(r'\[\d+\]', '', clean_ans).strip()

    sentences = [s.strip() for s in re.split(r'[.!?\n]', clean_ans) if len(s.strip()) > 15]
    if not sentences:
        return 1.0

    combined_context = " ".join(context_chunks).lower()
    embedder = _get_embedder()

    supported_count = 0
    for sentence in sentences:
        s_words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
        if not s_words:
            supported_count += 1
            continue

        # Lexical keyword match ratio
        overlap = sum(1 for w in s_words if w in combined_context) / len(s_words)
        if overlap >= 0.40:
            supported_count += 1
        else:
            # Semantic fallback check with embedder
            s_vec = embedder.embed_query(sentence)
            max_sim = 0.0
            for ctx in context_chunks:
                ctx_vec = embedder.embed_query(ctx[:400])
                sim = _cosine_similarity(s_vec, ctx_vec)
                if sim > max_sim:
                    max_sim = sim
            if max_sim >= 0.65:
                supported_count += 1

    return round(min(1.0, supported_count / len(sentences)), 4)


def compute_answer_relevance(question: str, answer: str) -> float:
    """
    Measures Answer Relevance (0.0 to 1.0):
    Computes semantic alignment between question and answer using cosine similarity.
    """
    if not question or not answer:
        return 0.0

    # If it's a safe refusal on an unanswerable question, relevance is high (safe behavior)
    if "don't have enough information" in answer.lower() or "could not find" in answer.lower():
        return 0.95

    clean_ans = re.sub(r'📚 Sources & Citations:.*', '', answer, flags=re.DOTALL).strip()
    embedder = _get_embedder()

    q_vec = embedder.embed_query(question)
    a_vec = embedder.embed_query(clean_ans)
    sim = _cosine_similarity(q_vec, a_vec)

    # Normalize similarity (MiniLM cosine range is roughly 0.2 to 0.9)
    normalized = max(0.0, min(1.0, (sim - 0.15) / 0.75))
    return round(float(normalized), 4)


def compute_context_precision(ground_truth: str, retrieved_chunks: List[dict]) -> float:
    """
    Measures Context Precision (0.0 to 1.0):
    Evaluates whether relevant passages are ranked higher in the retrieved chunk list.
    """
    if not ground_truth or not retrieved_chunks:
        return 1.0 if not retrieved_chunks else 0.0

    embedder = _get_embedder()
    gt_vec = embedder.embed_query(ground_truth)

    precision_scores = []
    hits = 0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_text = chunk.get("text", "")
        c_vec = embedder.embed_query(chunk_text[:300])
        sim = _cosine_similarity(gt_vec, c_vec)

        # A chunk is considered relevant if similarity > 0.50
        is_relevant = sim >= 0.50
        if is_relevant:
            hits += 1
            precision_scores.append(hits / rank)

    if not precision_scores:
        # If ground truth was empty or an unanswerable question
        return 1.0 if "don't have enough" in ground_truth.lower() else 0.0

    return round(float(sum(precision_scores) / len(precision_scores)), 4)


def compute_context_recall(ground_truth: str, retrieved_chunks: List[dict]) -> float:
    """
    Measures Context Recall (0.0 to 1.0):
    Evaluates what proportion of ground truth facts are captured by the retrieved context.
    """
    if not ground_truth:
        return 1.0

    if "don't have enough" in ground_truth.lower():
        return 1.0

    if not retrieved_chunks:
        return 0.0

    combined_context = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
    gt_keywords = set(re.findall(r'\b\w{4,}\b', ground_truth.lower()))

    if not gt_keywords:
        return 1.0

    found_count = sum(1 for w in gt_keywords if w in combined_context)
    return round(float(found_count / len(gt_keywords)), 4)


def evaluate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs an end-to-end RAG query for a test sample and calculates all 4 RAGAS metrics.
    """
    question = sample["question"]
    ground_truth = sample.get("ground_truth", "")
    expected_sources = sample.get("expected_sources", [])

    # 1. Guardrail Safety Check
    is_safe, safety_msg = check_input_safety(question)
    if not is_safe:
        return {
            "id": sample.get("id", "sample"),
            "question": question,
            "answer": safety_msg,
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
            "ragas_score": 1.0,
            "guardrail_status": "BLOCKED",
        }

    # 2. Intent Routing
    intent_reply = detect_conversational_intent(question)
    if intent_reply is not None:
        return {
            "id": sample.get("id", "sample"),
            "question": question,
            "answer": intent_reply,
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
            "ragas_score": 1.0,
            "guardrail_status": "FAST_ROUTED",
        }

    # 3. Retrieval & Reranking
    raw_chunks = retrieve_chunks(question, k=TOP_K)
    ranked_chunks = rerank_chunks(question, raw_chunks, top_n=RERANK_TOP_N) if raw_chunks else []
    is_confident, qualified_chunks = validate_retrieval_confidence(ranked_chunks)

    if not is_confident or not qualified_chunks:
        refusal = safe_refusal_response()
        f_score = 1.0 if not expected_sources else 0.2
        r_score = compute_answer_relevance(question, refusal)
        p_score = 1.0 if not expected_sources else 0.0
        rec_score = 1.0 if not expected_sources else 0.0
        ragas_avg = round(np.mean([f_score, r_score, p_score, rec_score]), 4)
        return {
            "id": sample.get("id", "sample"),
            "question": question,
            "answer": refusal,
            "faithfulness": f_score,
            "answer_relevance": r_score,
            "context_precision": p_score,
            "context_recall": rec_score,
            "ragas_score": ragas_avg,
            "guardrail_status": "SAFE_REFUSAL",
        }

    # 4. LLM Generation & Output Guardrail
    prompt = build_prompt(question, qualified_chunks)
    raw_answer = get_answer(prompt)
    is_safe_output, answer, alert_msg = validate_output_safety(raw_answer)
    citations, footer = resolve_citations(answer, qualified_chunks)
    full_answer = f"{answer}{footer}"

    # 5. Compute RAGAS Metrics
    context_texts = [c.get("parent_text", c.get("text", "")) for c in qualified_chunks]
    faithfulness = compute_faithfulness(answer, context_texts)
    relevance = compute_answer_relevance(question, answer)
    precision = compute_context_precision(ground_truth, qualified_chunks)
    recall = compute_context_recall(ground_truth, qualified_chunks)
    ragas_score = round(float(np.mean([faithfulness, relevance, precision, recall])), 4)

    return {
        "id": sample.get("id", "sample"),
        "question": question,
        "ground_truth": ground_truth,
        "answer": full_answer,
        "retrieved_sources": [c.get("source") for c in qualified_chunks],
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_precision": precision,
        "context_recall": recall,
        "ragas_score": ragas_score,
        "guardrail_status": "ANSWERED",
    }


def evaluate_dataset(dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluates an entire benchmark dataset and generates an aggregate evaluation scorecard.
    """
    if dataset_path is None:
        dataset_path = "evaluation/datasets/benchmark_dataset.json"

    data_file = Path(dataset_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    samples = json.loads(data_file.read_text(encoding="utf-8"))
    results = []

    for i, sample in enumerate(samples, start=1):
        print(f"  [Evaluating {i}/{len(samples)}] '{sample['question'][:50]}...'")
        res = evaluate_sample(sample)
        results.append(res)

    # Compute aggregate mean scores
    avg_faithfulness = round(float(np.mean([r["faithfulness"] for r in results])), 4)
    avg_relevance = round(float(np.mean([r["answer_relevance"] for r in results])), 4)
    avg_precision = round(float(np.mean([r["context_precision"] for r in results])), 4)
    avg_recall = round(float(np.mean([r["context_recall"] for r in results])), 4)
    overall_ragas = round(float(np.mean([r["ragas_score"] for r in results])), 4)

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_samples": len(results),
        "aggregate_metrics": {
            "faithfulness": avg_faithfulness,
            "answer_relevance": avg_relevance,
            "context_precision": avg_precision,
            "context_recall": avg_recall,
            "ragas_score": overall_ragas,
        },
        "sample_evaluations": results,
    }

    # Save evaluation report
    out_dir = Path("data/eval_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"ragas_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Evaluation report saved to: {report_file}")

    return report
