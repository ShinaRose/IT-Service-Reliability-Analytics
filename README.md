# Reliability Analytics Platform

Turns synthetic alerts/incidents/deployments into DORA metrics, measured alert-noise
reduction, and a ranked list of where to spend engineering effort -- with an AI layer
that summarizes, categorizes and retrieves, but never computes a metric itself.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # or `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
pip install -e .
```

Python 3.13 was used in development (the spec calls for 3.12; nothing here is
version-specific beyond `>=3.11`).

## Pipeline, in order

```bash
# 1. Generate 12 months of synthetic data into data/reliability.duckdb
python -m relplatform.generator.cli --seed 42 --months 12

# 2. Cluster alerts (embeds messages via sentence-transformers, cached by content hash;
#    first run downloads all-MiniLM-L6-v2, ~90MB)
python scripts/run_clustering.py

# 3. Compute DORA metrics, change-failure model, MTTR fits, capacity forecast, risk
#    ranking, and persist the full report to DuckDB
python scripts/run_pipeline.py

# 4. Run the mandatory eval suite (clustering ARI/purity, root-cause accuracy vs
#    keyword baseline, retrieval precision@3, latency/token report)
python scripts/run_eval.py

# 5. Serve
uvicorn relplatform.api.main:app --reload --app-dir src
streamlit run src/relplatform/dashboard/app.py

# 6. Tests
pytest
```

## Deploying the live dashboard (Streamlit Community Cloud)

`docs/index.html` (published via GitHub Pages) is a frozen static snapshot. To make the
*interactive* dashboard -- recomputable, with a working "Generate exec summary" button --
reachable from a URL instead of just `localhost`, deploy it to
[Streamlit Community Cloud](https://share.streamlit.io) (free):

1. Go to share.streamlit.io, sign in with GitHub, click **New app**.
2. Pick this repo, branch `main`, and set the main file path to
   `src/relplatform/dashboard/app.py`.
3. Before deploying, open **Advanced settings -> Secrets** and paste the contents of
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) (edit the values
   first -- see the comments in that file). At minimum, set `RELPLATFORM_MONTHS = "3"`
   so first-run generation and embedding finish quickly on the free tier's 1 CPU / 1GB
   RAM; the full local setup defaults to 12 months.
4. Deploy. First load takes a few minutes -- the app has no data yet (the 145MB
   `reliability.duckdb` is git-ignored, over GitHub's 100MB limit) and bootstraps itself
   on first run: generates synthetic data, downloads the MiniLM embedding model,
   clusters alerts, and computes the report (`relplatform/bootstrap.py`). Every restart
   after that reuses what's already in the container until it's redeployed or recycled.
5. `RELPLATFORM_PROVIDER` defaults to `mock` if you skip the secrets step -- the app
   still works, the exec-summary button just returns placeholder text. `ollama` will
   not work on Streamlit Cloud (there's no Ollama server to reach from there); use
   `gemini` with a free-tier `GEMINI_API_KEY` for a real AI-generated exec summary in
   the cloud.

## AI provider

Selected via `RELPLATFORM_PROVIDER` env var (default `mock`):

| Value | Requirements |
|---|---|
| `ollama` | A local [Ollama](https://ollama.com) server with `llama3.2:3b` pulled (`ollama pull llama3.2:3b`) |
| `gemini` | `GEMINI_API_KEY` env var (free tier) |
| `mock` | Nothing -- deterministic offline provider for tests/CI, synthesizes schema-valid dummy JSON |

Embeddings (`ModelProvider.embed()`) always use local sentence-transformers
(`all-MiniLM-L6-v2`) regardless of the generation provider -- that's the zero-budget
piece that doesn't vary. See `relplatform/ai/provider.py`.

**The eval numbers in this repo's `data/eval_report.json` were produced with the `mock`
provider** (no Ollama/Gemini available in the dev sandbox). Mock's root-cause
categorization always returns the same JSON-schema-valid dummy category, so its
"accuracy" there is really just the majority-class rate -- it demonstrates the
plumbing (schema enforcement, caching, retry, batch reporting) end-to-end, not real
categorization quality. Point `RELPLATFORM_PROVIDER` at `ollama` or `gemini` and rerun
`scripts/run_eval.py` for a real accuracy number; the keyword baseline it's compared
against is real either way.

## Architecture

```
relplatform/
  generator/     synthetic data: services + dependency graph, deployments, incidents,
                 alert storms (varied wording, correlated via the dependency graph),
                 genuine multi-paragraph postmortems, resource metrics
  analytics/     clustering (dedup), dora (4 official metrics), change_failure
                 (logistic regression), mttr (log-normal/Weibull fit), capacity
                 (trend forecast), risk (composite ranking), embeddings (cache)
  ai/            provider.py (ModelProvider: generate/embed/structured_output),
                 cache.py (DuckDB content-hash cache), tasks.py (narrative,
                 root-cause), rag.py (retrieval), exec_summary.py, numeric_guard.py
  eval/          clustering_eval, root_cause_eval, retrieval_eval, latency_report
  api/, dashboard/  FastAPI service, Streamlit dashboard
scripts/         CLI entry points tying the above together
tests/           pytest suite, including the mandatory hallucination-guard test
labels/          root_cause_labels.csv -- the hand-label eval set (see below)
```

### Why the numbers are trustworthy, not just plausible

- **Change-failure ground truth is real, and calibrated to be learnable.** The generator
  decides per-deployment whether it triggers an incident via a logistic function of
  actual deploy features (size, commit count, weekend/off-hours, service baseline risk).
  Early calibration produced a *theoretical* ceiling AUC of 0.53 (Bayes-optimal score vs.
  realized outcome) -- essentially noise -- so it was recalibrated to a ceiling of ~0.67;
  the shipped model's 5-fold CV AUC is ~0.64. See the comment in
  `relplatform/generator/simulate.py::_deploy_risk`.
- **Clustering ground truth is held out from the algorithm.** `alerts.incident_id` is
  never used as a clustering feature, only for `evaluate_against_ground_truth` (ARI +
  purity). Current numbers: ARI ≈ 0.79, purity ≈ 0.93, noise reduction ≈ 86.6%
  (41,572 raw alerts → 5,550 distinct clusters, of which 754 are real multi-alert
  groups and the rest are singletons DBSCAN correctly left as noise).
- **RAG retrieval doesn't leak the label, and matches like-to-like.** The incident index
  and query text deliberately exclude `root_cause_category` -- an early version embedded
  it directly into both, which made precision@3 measure keyword matching (0.97) instead
  of semantic similarity. Fixing that (querying with the incident's own alert-cluster
  text against an index of postmortem prose) dropped precision@3 to 0.17 -- *worse* than
  the 0.27 you'd get from guessing by category frequency alone, because alert text
  (short, templated, numeric) and postmortem prose are different enough in style that a
  general sentence embedding doesn't bridge them, and a full postmortem runs well past
  MiniLM's ~256-token window so most of it was silently dropped anyway. The index is now
  built from each historical incident's own alert-storm text instead (like-to-like;
  postmortem text is still used for the displayed resolution snippet, just not for
  similarity) -- precision@3 ≈ 0.69. This also exposed a real generator gap: alert
  *signal types* (cpu/memory/timeout/etc.) weren't correlated with root cause category
  at all, so there was nothing genuine to retrieve on regardless of text choice. Fixed
  via `CATEGORY_SIGNAL_WEIGHTS` in `relplatform/generator/alert_messages.py` (a
  resource_exhaustion incident's storm is now really dominated by cpu/memory/
  connection_pool alerts, the way a real one would be). See `relplatform/ai/rag.py`.
- **Capacity forecasting requires a statistically real trend, not just a positive
  slope.** With ~365 days of noisy daily data, a linear fit finds a technically-positive
  slope for almost every service by chance, and with that much sample size some of those
  reach p < 0.05 significance despite r² as low as 0.02 -- pure noise dressed up as a
  confident-looking breach date (several services initially showed a "projected breach"
  sitting exactly at the forecast horizon cap). `forecast_service` now requires p < 0.05
  *and* r² ≥ 0.10 before reporting a `breach_projected` date; everything else is
  `no_significant_trend` or `stable_or_declining`. Only the two services the generator
  actually gives an upward trend (`payments-service`, r²≈0.90; `checkout-service`,
  r²≈0.79) come out as `breach_projected`. See `relplatform/analytics/capacity.py`.
- **The exec summary cannot state a number that isn't in its input.** Enforced by
  `relplatform/ai/numeric_guard.py` (tolerant of reformatting like "7%" for "7.29%", but
  not of numbers with no source), tested in `tests/test_hallucination.py`.

### Hand-labeled postmortems

"Hand-labeling" 100 postmortems for the root-cause eval uses the generator's own
`root_cause_category` for a stratified sample -- for synthetic data this plays the role a
human labeler would: it's the fixed answer key the postmortem text was generated to be
consistent with, independent of what any classifier looks at. `labels/root_cause_labels.csv`
is a static, inspectable eval set regenerated by `scripts/run_eval.py`.

### DORA band caveats

The published DORA bands (`relplatform/config.py::DORA_BANDS`,
`relplatform/analytics/dora.py`) are reproduced as-is including two known
discontinuities in the source report: `change_failure_rate`'s High and Medium bands are
both "16-30%", and `time_to_restore` has a gap between "one week" (Medium) and "six
months" (Low). Not a bug here -- the official chart really has this shape.
