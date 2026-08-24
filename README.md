# Reliability Analytics Platform

[![tests](https://github.com/ShinaRose/IT-Service-Reliability-Analytics/actions/workflows/tests.yml/badge.svg)](https://github.com/ShinaRose/IT-Service-Reliability-Analytics/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Report](https://img.shields.io/badge/report-live-2DD4BF)](https://shinarose.github.io/IT-Service-Reliability-Analytics/)

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

**Root-cause categorization in `data/eval_report.json` was rerun with a real provider**
(`ollama` / `llama3.2:3b`, local CPU inference): **87% accuracy** on the 100
hand-labeled postmortems, vs. 52% for the keyword baseline -- 100% of responses were
valid, schema-conforming JSON on the first attempt, at 31.6s / 41.6s per call (p50/p95).
An earlier run with the `mock` provider (no Ollama/Gemini available yet) scored 46%,
which was never a real number -- Mock's root-cause categorization always returns the
same dummy category, so that 46% was just the majority-class rate, demonstrating the
plumbing (schema enforcement, caching, retry, batch reporting) rather than actual
categorization quality. `mock` still is what a fresh clone gets by default (zero
external dependencies), and every other eval number here (clustering, retrieval) is
provider-independent -- only root-cause categorization needed a real model to mean
anything.

## What's real, what's synthetic, what this can't do

Stated plainly, in one place, rather than scattered through the numbered phases above.

**Synthetic by default.** Everything on the Home page and the SLOs/Financial/On-Call/
Structural/Evaluation tabs runs on `generator/simulate.py`'s output -- 12 months of
services, deployments, incidents, alerts, and an on-call roster, generated from a fixed
`seed` (default 42, `RELPLATFORM_SEED` env var to change it; the current seed is shown
in the Home page sidebar). Deterministic given the same seed *and* the same calendar
day -- `simulate()` anchors its 12-month window to "now," so the same seed run on a
different day produces a different but internally-consistent dataset. That's expected,
not a bug: the point of the seed is reproducibility of the generation *logic*, not a
byte-identical file forever.

**Real, when you ask for it.** The Real-World DORA page (Phase 5) computes the same
four DORA metrics from either a real public GitHub repo (releases, commits, and
labeled issues via the GitHub REST API) or your own uploaded CSVs. That page always
shows which of the three sources -- synthetic, GitHub-derived, or user-uploaded --
produced whatever's on screen; nothing else in the app currently reads real data.

**What the models assume:**
- Every euro figure (Financial Impact, the What-If Sandbox) rests on `config/costs.yaml`
  -- illustrative example rates, not measured business figures. Change-failure
  attribution uses a time-proximity heuristic (nearest prior deploy within a window),
  not a ground-truth causal trace, for both synthetic and uploaded data.
- The change-failure model, MTTR fits, capacity forecasts, and structural analytics
  (blast radius, propagation mining, change-point detection, reliability curves) are all
  plain, inspectable statistical/ML methods -- logistic regression, distribution fitting,
  linear trend tests, CUSUM, Kaplan-Meier. No LLM ever produces a number anywhere in this
  app; the AI layer only summarizes, categorizes, and retrieves text, and every number an
  LLM-generated summary states is checked against the real report by
  `relplatform/ai/numeric_guard.py` before it reaches the screen.
- Where a computation rests on a real, non-obvious assumption, the UI says so next to the
  number (`theme.assumption_note`) rather than only in a docstring.

**What this can't do:**
- It doesn't know your actual infrastructure, on-call rotation, or cost structure --
  the synthetic path is a demonstration of the analytics, not a fit to any real
  organization's numbers, and the "your own data" paths are only as good as what you map
  into them.
- The GitHub connector approximates (release = deployment, commit-time bucketing for
  lead time, revert-commit detection for change failure) -- see the Real-World DORA
  page's own assumption note for the specifics and their limits.
- The change-point detector, propagation miner, and blast-radius model are all evaluated
  against this platform's own synthetic ground truth (see the Evaluation tab) or the
  real dependency graph, not against a real production incident history -- their
  reported precision/recall/lead-time numbers describe how well they recover *this*
  generator's structure, not a guarantee about any other system.

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
  eval/          clustering_eval, root_cause_eval, retrieval_eval, calibration_eval,
                 latency_report -- backs the Evaluation dashboard tab (Phase 6),
                 regenerated by scripts/run_eval.py into data/eval_report.json
  slo/           error budgets, multi-window multi-burn-rate alerting (Phase 1)
  finance/       incident/toil cost, DORA-band counterfactuals, risk-vs-euro re-rank,
                 the What-If Sandbox's continuous-slider math (Phases 2 and 7)
  oncall/        pages-per-shift percentiles, out-of-hours/sleep-hours load, interrupt
                 concentration, alert fatigue score (Phase 3)
  structural/    dependency-graph blast radius/criticality, failure-propagation mining,
                 CUSUM change-point detection, Kaplan-Meier reliability curves (Phase 4)
  external/      GitHub REST API DORA connector + bring-your-own-data CSV mapping
                 (Phase 5) -- the only place real (non-synthetic) data enters the app
  reporting/     one-page PDF executive summary (Phase 7)
  api/, dashboard/  FastAPI service, Streamlit dashboard (one page per phase under
                 dashboard/pages/, lazy-loaded -- Streamlit only runs the page in view)
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
