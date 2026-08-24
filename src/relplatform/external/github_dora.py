"""GitHub REST API connector: computes genuine DORA metrics for any public repository,
reusing the exact same banding/trend functions (relplatform.analytics.dora) the
synthetic pipeline uses -- so a GitHub-derived Elite/High/Medium/Low band means the same
thing here as it does on the synthetic dashboard, not a second scoring scheme.

Every mapping from "what GitHub's API actually returns" to "one of the four DORA
metrics" is a real judgment call, documented inline where the call is made, not left
implicit:

- Deployment = a published GitHub Release. Repos with zero releases have no
  deployment-frequency signal from this connector -- deliberately no silent fallback to
  counting merge commits, which conflates "code merged" with "code deployed".
- Lead time for a release = release.published_at - the EARLIEST default-branch commit
  timestamp strictly after the previous release and at or before this one. An
  approximation of "first commit to deploy" using commit-time bucketing, not a precise
  per-PR trace; the first release in a repo's history has no prior release to bucket
  from and is excluded from the lead-time calculation (still counted for frequency).
- Change failure = a commit whose message starts with "Revert" (GitHub's own default
  title for a revert PR) falling within a release's commit bucket. This attributes the
  revert to whichever release's time window it lands in, not necessarily the release
  that introduced the reverted change.
- Time to restore = closed_at - created_at for issues carrying a user-specified incident
  label. There is no universal label convention across repos, so it is never guessed --
  the caller supplies one, or this metric is reported unavailable.

Unauthenticated GitHub REST API calls are capped at 60/hour per IP. This connector: (1)
caches every raw response in DuckDB's `github_cache` table, so repeat views of the same
repo within the TTL cost zero API calls; (2) bounds calls per repo view to a small,
capped number of paginated requests (releases + commits + optionally issues); and (3)
raises a typed GitHubRateLimitError with the reset time on a 403/429 rate-limit
response, instead of letting an opaque HTTP error propagate -- callers show that
message, not a stack trace.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import pandas as pd

from relplatform.analytics.dora import change_failure_rate as _change_failure_rate_bands
from relplatform.analytics.dora import deployment_frequency as _deployment_frequency_bands
from relplatform.analytics.dora import lead_time_for_changes as _lead_time_bands
from relplatform.analytics.dora import time_to_restore as _time_to_restore_bands

API_ROOT = "https://api.github.com"
CACHE_TTL_MINUTES = 60.0
REPO_META_CACHE_TTL_MINUTES = 60.0 * 24  # default branch changes rarely
MAX_RELEASES = 30
COMMITS_PER_PAGE = 100
MAX_COMMIT_PAGES = 3          # up to 300 commits -- the connector's stated rate-limit/scope bound
ISSUES_PER_PAGE = 100
MAX_ISSUE_PAGES = 3
REQUEST_TIMEOUT_SECONDS = 15.0

_REVERT_PREFIX_RE = re.compile(r"^revert\b", re.IGNORECASE)


class GitHubRateLimitError(Exception):
    def __init__(self, reset_at: datetime | None, message: str):
        self.reset_at = reset_at
        super().__init__(message)


class GitHubRepoError(Exception):
    """Malformed repo reference, or the repo/resource wasn't found (404)."""


def parse_repo_input(text: str) -> tuple[str, str]:
    text = text.strip()
    text = re.sub(r"^https?://github\.com/", "", text)
    text = text.rstrip("/")
    parts = text.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise GitHubRepoError(f"Expected 'owner/repo' or a github.com URL, got: {text!r}")
    return parts[0], parts[1]


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def _get_cached(con, url: str, ttl_minutes: float):
    if con is None:
        return None
    row = con.execute(
        "SELECT response_json, fetched_at FROM github_cache WHERE cache_key = ?", [_cache_key(url)]
    ).fetchone()
    if row is None:
        return None
    response_json, fetched_at = row
    age_minutes = (datetime.now(timezone.utc) - pd.Timestamp(fetched_at).tz_localize(timezone.utc)).total_seconds() / 60
    if age_minutes > ttl_minutes:
        return None
    return json.loads(response_json)


def _store_cache(con, url: str, data) -> None:
    if con is None:
        return
    con.execute(
        "INSERT OR REPLACE INTO github_cache (cache_key, url, fetched_at, response_json) VALUES (?,?,?,?)",
        [_cache_key(url), url, datetime.now(timezone.utc).replace(tzinfo=None), json.dumps(data)],
    )


def _request(con, url: str, token: str | None, ttl_minutes: float):
    cached = _get_cached(con, url, ttl_minutes)
    if cached is not None:
        return cached

    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = httpx.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    except httpx.RequestError as e:
        raise GitHubRepoError(f"Network error reaching GitHub: {e}") from e

    if resp.status_code == 404:
        raise GitHubRepoError(f"Not found: {url} -- check the owner/repo name and that the repo is public.")
    if resp.status_code in (403, 429) and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset_raw = resp.headers.get("X-RateLimit-Reset")
        reset_at = datetime.fromtimestamp(int(reset_raw), tz=timezone.utc) if reset_raw else None
        raise GitHubRateLimitError(reset_at, (
            "GitHub API rate limit exhausted"
            + (f" -- resets at {reset_at.isoformat()}" if reset_at else "")
            + ". Set a GITHUB_TOKEN (env var or Streamlit secret) for a 5,000/hour limit instead of 60/hour."
        ))
    resp.raise_for_status()
    data = resp.json()
    _store_cache(con, url, data)
    return data


def _paginated(con, url_base: str, token: str | None, per_page: int, max_pages: int, ttl_minutes: float) -> list[dict]:
    results: list[dict] = []
    for page in range(1, max_pages + 1):
        sep = "&" if "?" in url_base else "?"
        url = f"{url_base}{sep}per_page={per_page}&page={page}"
        data = _request(con, url, token, ttl_minutes)
        if not isinstance(data, list) or len(data) == 0:
            break
        results.extend(data)
        if len(data) < per_page:
            break
    return results


def fetch_default_branch(con, owner: str, repo: str, token: str | None = None) -> str:
    data = _request(con, f"{API_ROOT}/repos/{owner}/{repo}", token, REPO_META_CACHE_TTL_MINUTES)
    if "default_branch" not in data:
        raise GitHubRepoError(f"Could not read repo metadata for {owner}/{repo}: {data.get('message', data)}")
    return data["default_branch"]


def fetch_releases(con, owner: str, repo: str, token: str | None = None) -> list[dict]:
    releases = _paginated(
        con, f"{API_ROOT}/repos/{owner}/{repo}/releases", token,
        per_page=min(MAX_RELEASES, 100), max_pages=1, ttl_minutes=CACHE_TTL_MINUTES,
    )
    return releases[:MAX_RELEASES]


def fetch_commits(con, owner: str, repo: str, branch: str, since: pd.Timestamp | None, token: str | None = None) -> list[dict]:
    url = f"{API_ROOT}/repos/{owner}/{repo}/commits?sha={branch}"
    if since is not None:
        url += f"&since={since.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    return _paginated(con, url, token, per_page=COMMITS_PER_PAGE, max_pages=MAX_COMMIT_PAGES, ttl_minutes=CACHE_TTL_MINUTES)


def fetch_issues(con, owner: str, repo: str, label: str, token: str | None = None) -> list[dict]:
    url = f"{API_ROOT}/repos/{owner}/{repo}/issues?state=all&labels={label}"
    issues = _paginated(con, url, token, per_page=ISSUES_PER_PAGE, max_pages=MAX_ISSUE_PAGES, ttl_minutes=CACHE_TTL_MINUTES)
    return [i for i in issues if "pull_request" not in i]  # the issues endpoint also returns PRs


def _is_revert_commit(message: str) -> bool:
    return bool(_REVERT_PREFIX_RE.match(message.strip()))


def build_deployments_df(releases: list[dict], commits: list[dict]) -> tuple[pd.DataFrame, str]:
    """One row per release with a real publish date: deployed_at, lead_time_hours (NaN
    if this is the first release or no fetched commit falls in its bucket),
    caused_incident (1 if a revert-message commit falls in its bucket)."""
    cols = ["deployed_at", "lead_time_hours", "caused_incident"]
    releases = sorted((r for r in releases if r.get("published_at")), key=lambda r: r["published_at"])
    if not releases:
        return pd.DataFrame(columns=cols), (
            "No releases found for this repo -- deployment frequency and lead time are "
            "unavailable. This connector does not fall back to counting merge commits "
            "(that conflates 'code merged' with 'code deployed')."
        )

    commits = sorted(
        (c for c in commits if c.get("commit", {}).get("committer", {}).get("date")),
        key=lambda c: c["commit"]["committer"]["date"],
    )
    commit_times = [pd.Timestamp(c["commit"]["committer"]["date"]).tz_localize(None) for c in commits]
    commit_messages = [c["commit"]["message"] for c in commits]

    rows = []
    n_excluded_from_lead_time = 0
    prev_published = None
    for r in releases:
        published_at = pd.Timestamp(r["published_at"]).tz_localize(None)
        bucket_idx = [i for i, t in enumerate(commit_times) if (prev_published is None or t > prev_published) and t <= published_at]

        lead_time_hours = None
        if prev_published is not None and bucket_idx:
            earliest = min(commit_times[i] for i in bucket_idx)
            lead_time_hours = (published_at - earliest).total_seconds() / 3600
        elif prev_published is not None:
            n_excluded_from_lead_time += 1

        caused_incident = 1 if any(_is_revert_commit(commit_messages[i]) for i in bucket_idx) else 0
        rows.append({"deployed_at": published_at, "lead_time_hours": lead_time_hours, "caused_incident": caused_incident})
        prev_published = published_at

    note = (
        f"{len(rows)} releases found. Lead time excludes the first release (no prior "
        f"release to bucket commits from) and {n_excluded_from_lead_time} more release(s) "
        f"whose commit bucket had no fetched commits (only the most recent "
        f"{MAX_COMMIT_PAGES * COMMITS_PER_PAGE} default-branch commits since the oldest "
        f"release are fetched, to bound API calls -- older or high-volume history may be "
        f"undercounted)."
    )
    return pd.DataFrame(rows), note


def build_incidents_df(issues: list[dict]) -> tuple[pd.DataFrame, str]:
    cols = ["started_at", "resolved_at"]
    if not issues:
        return pd.DataFrame(columns=cols), "No issues found with the given label -- time to restore is unavailable."

    rows = []
    n_open = 0
    for i in issues:
        if i.get("state") != "closed" or not i.get("closed_at"):
            n_open += 1
            continue
        rows.append({
            "started_at": pd.Timestamp(i["created_at"]).tz_localize(None),
            "resolved_at": pd.Timestamp(i["closed_at"]).tz_localize(None),
        })
    note = f"{len(rows)} closed incident-labeled issues used; {n_open} still open and excluded (no resolution time yet)."
    return pd.DataFrame(rows, columns=cols), note


@dataclass
class GitHubDoraResult:
    owner: str
    repo: str
    default_branch: str
    deployment_frequency: dict | None
    lead_time_for_changes: dict | None
    change_failure_rate: dict | None
    time_to_restore: dict | None
    notes: list[str] = field(default_factory=list)
    fetched_at: str = ""


def compute_github_dora(
    owner: str, repo: str, incident_label: str | None = None, con=None, token: str | None = None,
) -> GitHubDoraResult:
    notes: list[str] = []
    branch = fetch_default_branch(con, owner, repo, token)
    releases = fetch_releases(con, owner, repo, token)

    commits: list[dict] = []
    if releases:
        published_dates = [pd.Timestamp(r["published_at"]).tz_localize(None) for r in releases if r.get("published_at")]
        if published_dates:
            commits = fetch_commits(con, owner, repo, branch, since=min(published_dates), token=token)

    deployments_df, dep_note = build_deployments_df(releases, commits)
    notes.append(dep_note)

    deploy_freq = lead_time = change_failure = None
    if len(deployments_df):
        deploy_freq = _deployment_frequency_bands(deployments_df)
        change_failure = _change_failure_rate_bands(deployments_df)

        lt_input = deployments_df.dropna(subset=["lead_time_hours"])
        if len(lt_input):
            lead_time = _lead_time_bands(lt_input)
        else:
            notes.append("No release had a computable lead time -- lead time for changes is unavailable.")

    time_to_restore = None
    if not incident_label:
        notes.append("No incident label provided -- time to restore is unavailable. Supply the label this repo uses for incident/outage issues.")
    else:
        issues = fetch_issues(con, owner, repo, incident_label, token)
        incidents_df, inc_note = build_incidents_df(issues)
        notes.append(inc_note)
        if len(incidents_df):
            restore_hours = (pd.to_datetime(incidents_df["resolved_at"]) - pd.to_datetime(incidents_df["started_at"])).dt.total_seconds() / 3600
            if (restore_hours > 0).any():
                time_to_restore = _time_to_restore_bands(incidents_df)
            else:
                notes.append("All matched incident issues had zero or negative resolution time -- time to restore is unavailable.")

    return GitHubDoraResult(
        owner=owner, repo=repo, default_branch=branch,
        deployment_frequency=deploy_freq, lead_time_for_changes=lead_time,
        change_failure_rate=change_failure, time_to_restore=time_to_restore,
        notes=notes, fetched_at=datetime.now(timezone.utc).isoformat(),
    )
