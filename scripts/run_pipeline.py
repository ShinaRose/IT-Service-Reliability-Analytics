"""Computes the full analytics report and persists it to DuckDB (`reports` table) for
the API and dashboard to read. Run scripts/run_clustering.py first (or pass
--recompute-clusters) so this reuses the persisted alert_clusters instead of reclustering
on every pipeline run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relplatform.db import get_connection
from relplatform.pipeline import compute_full_report, persist_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-clusters", action="store_true")
    args = ap.parse_args()

    con = get_connection()
    report = compute_full_report(con, use_cached_clusters=not args.recompute_clusters)
    persist_report(con, report)

    print("DORA bands:", {k: v["band"] for k, v in report["dora_metrics"].items()})
    print("Change failure model CV AUC:", report["change_failure_model"]["metrics"].get("cv_roc_auc_mean"))
    print("Noise reduction:", report["noise_reduction"]["noise_reduction_rate"])
    print("Clustering ARI/purity:", report["clustering_ground_truth_eval"])
    print("Top risk services:", [r["service"] for r in report["risk_scores"][:3]])
    con.close()


if __name__ == "__main__":
    main()
