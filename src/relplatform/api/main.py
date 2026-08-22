"""FastAPI service exposing the analytics report + AI-assisted endpoints.

Run with: uvicorn relplatform.api.main:app --reload
The heavy analytics (DORA, risk, MTTR, capacity, clustering) are precomputed by
scripts/run_pipeline.py and read from the `reports` table here; only the RAG lookup and
exec summary are computed live (and cached by content hash regardless).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from relplatform.ai.exec_summary import generate_exec_summary
from relplatform.ai.provider import get_provider
from relplatform.ai.rag import build_incident_index, retrieve_for_alert_cluster
from relplatform.db import get_connection
from relplatform.pipeline import compute_full_report, load_latest_report

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["con"] = get_connection()
    _state["provider"] = get_provider()
    yield
    _state["con"].close()


app = FastAPI(title="Reliability Analytics API", lifespan=lifespan)


def _report() -> dict:
    report = load_latest_report(_state["con"])
    if report is None:
        raise HTTPException(status_code=503, detail="No report computed yet -- run scripts/run_pipeline.py first")
    return report


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/report")
def full_report():
    return _report()


@app.get("/report/recompute")
def recompute_report():
    report = compute_full_report(_state["con"])
    from relplatform.pipeline import persist_report

    persist_report(_state["con"], report)
    return {"computed_at": report["computed_at"]}


@app.get("/dora")
def dora_metrics():
    return _report()["dora_metrics"]


@app.get("/change-failure")
def change_failure():
    return _report()["change_failure_model"]


@app.get("/mttr")
def mttr():
    return _report()["mttr_fits"]


@app.get("/capacity")
def capacity():
    return _report()["capacity_forecasts"]


@app.get("/risk")
def risk():
    return _report()["risk_scores"]


@app.get("/noise-reduction")
def noise_reduction():
    return _report()["noise_reduction"]


@app.get("/clustering-eval")
def clustering_eval():
    return _report()["clustering_ground_truth_eval"]


@app.get("/incidents/{incident_id}/similar")
def similar_incidents(incident_id: str, k: int = 3):
    con = _state["con"]
    incidents = con.execute("SELECT * FROM incidents").df()
    if incident_id not in incidents["id"].values:
        raise HTTPException(status_code=404, detail=f"unknown incident {incident_id}")

    all_alerts = con.execute("SELECT * FROM alerts").df()
    query_alerts = all_alerts[all_alerts["incident_id"] == incident_id]
    if query_alerts.empty:
        raise HTTPException(status_code=404, detail=f"no alerts recorded for {incident_id}")

    index = build_incident_index(con, incidents, all_alerts)
    results = retrieve_for_alert_cluster(con, index, query_alerts, k=k, exclude_incident_id=incident_id)
    return {"incident_id": incident_id, "similar_incidents": results}


@app.get("/exec-summary")
def exec_summary(period_label: str = "latest"):
    report = _report()
    con = _state["con"]
    provider = _state["provider"]
    risk_df = pd.DataFrame(report["risk_scores"])
    text, context, stats = generate_exec_summary(
        con, provider, report["dora_metrics"], report["noise_reduction"], risk_df,
        report["capacity_forecasts"], period_label,
    )
    return {"summary": text, "provider": provider.name, "model": provider.model,
            "tokens_in": stats.tokens_in, "tokens_out": stats.tokens_out, "cache_hit": stats.hit}
