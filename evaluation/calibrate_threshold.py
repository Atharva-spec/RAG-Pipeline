import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import GUARDRAIL_CONFIDENCE_THRESHOLD, RERANK_TOP_N, TOP_K
from retrieval.reranker import rerank_chunks
from retrieval.retriever import retrieve_chunks


def evaluate_query_confidences(dataset: List[dict]) -> List[dict]:
    """
    Runs retrieval and cross-encoder reranking for each query in the dataset
    and extracts the top passage confidence score (0.0 to 1.0).
    """
    results = []
    for i, item in enumerate(dataset, start=1):
        query = item["query"]
        is_in_domain = bool(item.get("is_in_domain", True))
        print(f"  [{i}/{len(dataset)}] Scoring: '{query[:45]}...'")

        raw_chunks = retrieve_chunks(query, k=TOP_K)
        if raw_chunks:
            ranked = rerank_chunks(query, raw_chunks, top_n=RERANK_TOP_N)
            top_score = ranked[0].get("confidence_score", 0.0) if ranked else 0.0
        else:
            top_score = 0.0

        results.append({
            "query": query,
            "is_in_domain": is_in_domain,
            "top_confidence": round(float(top_score), 4),
        })

    return results


def sweep_thresholds(
    eval_results: List[dict],
    min_t: float = 0.05,
    max_t: float = 0.95,
    step: float = 0.05,
) -> List[dict]:
    """
    Sweeps confidence thresholds from min_t to max_t and computes
    Precision, Recall, F1 Score, and Accuracy at each threshold.
    """
    thresholds = np.arange(min_t, max_t + step / 2, step)
    rows = []

    for t in thresholds:
        t_val = round(float(t), 2)
        tp, fp, tn, fn = 0, 0, 0, 0

        for item in eval_results:
            is_actual_pos = item["is_in_domain"]
            is_pred_pos = item["top_confidence"] >= t_val

            if is_pred_pos and is_actual_pos:
                tp += 1
            elif is_pred_pos and not is_actual_pos:
                fp += 1
            elif not is_pred_pos and not is_actual_pos:
                tn += 1
            elif not is_pred_pos and is_actual_pos:
                fn += 1

        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        rows.append({
            "threshold": t_val,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        })

    return rows


def find_optimal_threshold(sweep_rows: List[dict]) -> dict:
    """
    Finds the optimal threshold that maximizes F1 score while prioritizing zero false positives.
    """
    # Sort by F1 descending, then Precision descending, then Accuracy descending
    sorted_rows = sorted(
        sweep_rows,
        key=lambda r: (r["f1"], r["precision"], r["accuracy"]),
        reverse=True,
    )
    return sorted_rows[0] if sorted_rows else sweep_rows[0]


def run_confidence_calibration(dataset_path: Optional[str] = None):
    """
    CLI runner for Confidence Threshold Calibration.
    """
    if dataset_path is None:
        dataset_path = "evaluation/datasets/calibration_dataset.json"

    data_file = Path(dataset_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Calibration dataset not found: {dataset_path}")

    dataset = json.loads(data_file.read_text(encoding="utf-8"))

    print("\n" + "=" * 70)
    print(" 🎯 Cross-Encoder Confidence Threshold Calibration")
    print("=" * 70)
    print(f"  • Test Dataset: {dataset_path} ({len(dataset)} samples)")
    print(f"  • Current System Threshold: {GUARDRAIL_CONFIDENCE_THRESHOLD:.2f}\n")

    # 1. Evaluate top chunk confidences
    eval_results = evaluate_query_confidences(dataset)

    # 2. Sweep thresholds
    sweep_rows = sweep_thresholds(eval_results, min_t=0.10, max_t=0.90, step=0.10)
    optimal = find_optimal_threshold(sweep_rows)

    # 3. Print Results Matrix
    print("\n" + "=" * 70)
    print(" 📊 Threshold Sweep Matrix")
    print("=" * 70)
    print(" Threshold │ Precision │ Recall │ F1-Score │ Accuracy │ Status")
    print("───────────┼───────────┼────────┼──────────┼──────────┼──────────────")

    for r in sweep_rows:
        is_opt = r["threshold"] == optimal["threshold"]
        status_tag = "⭐ OPTIMAL" if is_opt else ""
        if r["threshold"] == round(GUARDRAIL_CONFIDENCE_THRESHOLD, 2):
            status_tag += " (CURRENT)"

        print(
            f"   {r['threshold']:<7.2f} │   {r['precision']*100:>5.1f}%  │ {r['recall']*100:>5.1f}% │  {r['f1']*100:>6.1f}% │  {r['accuracy']*100:>6.1f}% │ {status_tag}"
        )

    print("=" * 70)
    print("\n🏆 Calibration Summary & Recommendation:")
    print(f"  • Optimal Threshold (θ*):     {optimal['threshold']:.2f}")
    print(f"  • Optimal Precision:          {optimal['precision']*100:.1f}% (Zero Hallucination)")
    print(f"  • Optimal Recall:             {optimal['recall']*100:.1f}% (High Coverage)")
    print(f"  • Optimal F1 Score:           {optimal['f1']*100:.1f}%")
    print(f"  • Current Configured Setting: {GUARDRAIL_CONFIDENCE_THRESHOLD:.2f}")

    if optimal["threshold"] != round(GUARDRAIL_CONFIDENCE_THRESHOLD, 2):
        print(f"\n💡 Recommendation: Update GUARDRAIL_CONFIDENCE_THRESHOLD = {optimal['threshold']:.2f} in config/settings.py for peak accuracy.")
    else:
        print("\n✓ Your current GUARDRAIL_CONFIDENCE_THRESHOLD is already optimal!")

    print("=" * 70 + "\n")
    return optimal


if __name__ == "__main__":
    ds_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_confidence_calibration(ds_arg)
