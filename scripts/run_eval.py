"""Runs the mandatory eval suite: clustering ARI/purity, root-cause accuracy (AI vs
keyword baseline) on a hand-labeled sample, RAG retrieval precision@k, and a
latency/token report for every AI batch call made along the way. Writes:
  - labels/root_cause_labels.csv  (the fixed hand-label eval set)
  - data/eval_report.json         (full results)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relplatform.ai.cache import CachedCallStats
from relplatform.ai.provider import get_provider
from relplatform.ai.tasks import categorize_postmortem
from relplatform.config import LABELS_DIR
from relplatform.db import get_connection
from relplatform.eval.clustering_eval import run_clustering_eval
from relplatform.eval.latency_report import build_batch_report
from relplatform.eval.retrieval_eval import precision_at_k
from relplatform.eval.root_cause_eval import build_hand_label_sample, evaluate_ai_categorization, evaluate_keyword_baseline


def main():
    con = get_connection()
    incidents = con.execute("SELECT * FROM incidents").df()
    alerts = con.execute("SELECT id, service, message, fired_at, incident_id FROM alerts").df()

    print("=== Clustering eval ===")
    t0 = time.time()
    clustering_result = run_clustering_eval(con, alerts)
    print(f"  ({time.time()-t0:.1f}s) noise_reduction={clustering_result['noise_reduction']['noise_reduction_rate']:.4f} "
          f"ARI={clustering_result['ground_truth_eval']['adjusted_rand_index']:.4f} "
          f"purity={clustering_result['ground_truth_eval']['purity']:.4f}")

    print("\n=== Root cause: hand-label sample ===")
    sample = build_hand_label_sample(incidents, n=100)
    LABELS_DIR.mkdir(exist_ok=True)
    sample[["id", "service", "severity", "root_cause_category", "postmortem_text"]].to_csv(
        LABELS_DIR / "root_cause_labels.csv", index=False
    )
    print(f"  wrote {len(sample)} labeled incidents to {LABELS_DIR / 'root_cause_labels.csv'}")
    print(f"  category distribution: {sample['root_cause_category'].value_counts().to_dict()}")

    baseline_result = evaluate_keyword_baseline(sample)
    print(f"  keyword baseline accuracy: {baseline_result['accuracy']:.3f}")

    provider = get_provider()
    print(f"  AI categorization using provider={provider.name} model={provider.model}")
    ai_preds = []
    stats: list[CachedCallStats] = []
    t0 = time.time()
    for row in sample.itertuples():
        data, s = categorize_postmortem(con, provider, row.postmortem_text, row.id)
        ai_preds.append(data.get("root_cause_category", "deployment_regression"))
        stats.append(s)
    wall = time.time() - t0
    ai_result = evaluate_ai_categorization(sample, ai_preds)
    root_cause_batch_report = build_batch_report("root_cause", stats, wall)
    print(f"  AI categorization accuracy: {ai_result['accuracy']:.3f}  (baseline: {baseline_result['accuracy']:.3f})")
    print(f"  batch report: {root_cause_batch_report.to_dict()}")

    print("\n=== RAG retrieval precision@3 ===")
    retrieval_result = precision_at_k(con, incidents, alerts, k=3, sample_n=60)
    print(f"  precision@3 = {retrieval_result['precision_at_k']:.3f} over {retrieval_result['n_queries']} queries")

    report = {
        "clustering": {
            "noise_reduction": clustering_result["noise_reduction"],
            "ground_truth_eval": clustering_result["ground_truth_eval"],
        },
        "root_cause": {
            "n_hand_labeled": len(sample),
            "category_distribution": sample["root_cause_category"].value_counts().to_dict(),
            "keyword_baseline_accuracy": baseline_result["accuracy"],
            "ai_accuracy": ai_result["accuracy"],
            "ai_provider": provider.name,
            "ai_model": provider.model,
            "batch_report": root_cause_batch_report.to_dict(),
        },
        "retrieval": {
            "precision_at_3": retrieval_result["precision_at_k"],
            "n_queries": retrieval_result["n_queries"],
        },
    }
    out_path = Path(__file__).resolve().parents[1] / "data" / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote eval report to {out_path}")
    con.close()


if __name__ == "__main__":
    main()
