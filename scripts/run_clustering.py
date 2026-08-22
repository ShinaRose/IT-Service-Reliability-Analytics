"""Runs alert clustering over the full dataset in DuckDB, prints noise reduction +
ARI/purity against ground truth, and persists cluster assignments to `alert_clusters`."""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relplatform.analytics.clustering import (
    ClusteringConfig,
    cluster_alerts,
    evaluate_against_ground_truth,
    noise_reduction_rate,
)
from relplatform.db import get_connection


def main():
    con = get_connection()
    alerts = con.execute("SELECT id, service, message, fired_at, incident_id FROM alerts").df()
    print(f"Loaded {len(alerts)} alerts")

    t0 = time.time()
    cfg = ClusteringConfig()
    clustered = cluster_alerts(con, alerts, cfg)
    print(f"Clustering took {time.time()-t0:.1f}s")

    stats = noise_reduction_rate(clustered)
    eval_stats = evaluate_against_ground_truth(clustered)
    print("Noise reduction:", stats)
    print("Ground-truth eval:", eval_stats)

    run_id = str(uuid.uuid4())[:8]
    out = clustered[["id", "cluster_id", "service"]].rename(columns={"id": "alert_id"})
    out.insert(0, "run_id", run_id)
    con.execute("DELETE FROM alert_clusters")
    con.register("out_df", out)
    con.execute("INSERT INTO alert_clusters SELECT * FROM out_df")
    print(f"Persisted cluster assignments (run_id={run_id})")


if __name__ == "__main__":
    main()
