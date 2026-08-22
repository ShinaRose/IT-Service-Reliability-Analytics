"""Central configuration: paths, seeds, and env-driven provider selection."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = str(DATA_DIR / "reliability.duckdb")

LABELS_DIR = ROOT / "labels"

RANDOM_SEED = int(os.environ.get("RELPLATFORM_SEED", "42"))

SIM_START_MONTHS_AGO = 12
SERVICES = [
    "api-gateway",
    "auth-service",
    "checkout-service",
    "payments-service",
    "inventory-service",
    "notification-service",
    "search-service",
    "recommendation-service",
]

# AI provider selection: "ollama" (default), "gemini", "mock"
MODEL_PROVIDER = os.environ.get("RELPLATFORM_PROVIDER", "mock").lower()
OLLAMA_MODEL = os.environ.get("RELPLATFORM_OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GEMINI_MODEL = os.environ.get("RELPLATFORM_GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# DORA thresholds (per DORA/Accelerate research program bands)
DORA_BANDS = {
    "deployment_frequency": {
        "elite": "on-demand (multiple per day)",
        "high": "between once per week and once per month",
        "medium": "between once per month and once every six months",
        "low": "fewer than once every six months",
    },
    "lead_time_for_changes": {
        "elite": "less than one hour",
        "high": "between one day and one week",
        "medium": "between one month and six months",
        "low": "more than six months",
    },
    "change_failure_rate": {
        "elite": "0-15%",
        "high": "16-30%",
        "medium": "16-30%",
        "low": "more than 30%",
    },
    "time_to_restore": {
        "elite": "less than one hour",
        "high": "less than one day",
        "medium": "between one day and one week",
        "low": "more than six months",
    },
}
