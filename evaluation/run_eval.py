import sys
from evaluation.evaluator import evaluate_dataset


def run_evaluation_cli(dataset_path: str = None):
    print("\n" + "=" * 70)
    print(" 📊 Running Local RAGAS Evaluation Suite")
    print("=" * 70)

    try:
        report = evaluate_dataset(dataset_path)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return

    metrics = report["aggregate_metrics"]
    samples = report["sample_evaluations"]

    print("\n" + "=" * 70)
    print(" 🏆 RAGAS Benchmark Scorecard Summary")
    print("=" * 70)
    print(f"  • Overall RAGAS Score:       {metrics['ragas_score'] * 100:.1f}%")
    print(f"  • Faithfulness (Grounding):  {metrics['faithfulness'] * 100:.1f}%")
    print(f"  • Answer Relevance:          {metrics['answer_relevance'] * 100:.1f}%")
    print(f"  • Context Precision:         {metrics['context_precision'] * 100:.1f}%")
    print(f"  • Context Recall:            {metrics['context_recall'] * 100:.1f}%")
    print("=" * 70)

    print("\n📋 Sample-by-Sample Results:")
    for s in samples:
        status_icon = "🟢" if s["guardrail_status"] == "ANSWERED" else "🛡️"
        print(f"\n  {status_icon} [{s['id']}] Q: {s['question']}")
        print(f"     Status: {s['guardrail_status']} | RAGAS Score: {s['ragas_score']*100:.1f}%")
        print(f"     Faithfulness: {s['faithfulness']*100:.0f}% | Relevance: {s['answer_relevance']*100:.0f}% | Precision: {s['context_precision']*100:.0f}% | Recall: {s['context_recall']*100:.0f}%")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation_cli(dataset_arg)
