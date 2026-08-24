from datetime import datetime, timezone

import httpx
import pandas as pd
import pytest

from relplatform.external import github_dora as gh


def _response(url: str, status_code: int = 200, json_body=None, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status_code, json=json_body if json_body is not None else [],
        headers=headers or {}, request=httpx.Request("GET", url),
    )


# ---------------- parse_repo_input ----------------

def test_parse_repo_input_owner_slash_repo():
    assert gh.parse_repo_input("octocat/Hello-World") == ("octocat", "Hello-World")


def test_parse_repo_input_full_url():
    assert gh.parse_repo_input("https://github.com/octocat/Hello-World/") == ("octocat", "Hello-World")


def test_parse_repo_input_malformed():
    with pytest.raises(gh.GitHubRepoError):
        gh.parse_repo_input("not-a-repo-ref")


# ---------------- revert detection ----------------

def test_is_revert_commit_matches_github_default_title():
    assert gh._is_revert_commit('Revert "Add flaky feature" (#42)')
    assert gh._is_revert_commit("revert lowercase title")


def test_is_revert_commit_does_not_match_unrelated_word():
    assert not gh._is_revert_commit("Reverting engine changes soon")  # no word boundary after "revert"
    assert not gh._is_revert_commit("Fix: revert changes from #10")   # doesn't start with revert


# ---------------- build_deployments_df ----------------

def _release(published_at: str) -> dict:
    return {"published_at": published_at, "tag_name": published_at}


def _commit(date: str, message: str) -> dict:
    return {"commit": {"committer": {"date": date}, "message": message}}


def test_build_deployments_df_no_releases():
    df, note = gh.build_deployments_df([], [])
    assert len(df) == 0
    assert "No releases found" in note


def test_build_deployments_df_first_release_excluded_from_lead_time():
    releases = [_release("2026-01-10T00:00:00Z")]
    commits = [_commit("2026-01-05T00:00:00Z", "Add feature")]
    df, note = gh.build_deployments_df(releases, commits)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["lead_time_hours"])


def test_build_deployments_df_lead_time_and_revert_detection():
    releases = [_release("2026-01-01T00:00:00Z"), _release("2026-01-10T00:00:00Z")]
    commits = [
        _commit("2025-12-30T00:00:00Z", "before first release, not in any bucket"),
        _commit("2026-01-05T00:00:00Z", "Add feature"),
        _commit("2026-01-08T00:00:00Z", 'Revert "Add feature" (#1)'),
    ]
    df, note = gh.build_deployments_df(releases, commits)
    assert len(df) == 2
    second = df.iloc[1]
    # earliest commit in the second release's bucket is 2026-01-05, published 2026-01-10 -> 120h
    assert second["lead_time_hours"] == pytest.approx(120.0)
    assert second["caused_incident"] == 1
    assert df.iloc[0]["caused_incident"] == 0


# ---------------- build_incidents_df ----------------

def test_build_incidents_df_no_issues():
    df, note = gh.build_incidents_df([])
    assert len(df) == 0
    assert "No issues found" in note


def test_build_incidents_df_excludes_open_issues():
    issues = [
        {"state": "closed", "created_at": "2026-01-01T00:00:00Z", "closed_at": "2026-01-01T02:00:00Z"},
        {"state": "open", "created_at": "2026-01-02T00:00:00Z", "closed_at": None},
    ]
    df, note = gh.build_incidents_df(issues)
    assert len(df) == 1
    assert "1 still open" in note


# ---------------- caching + rate limits (mocked HTTP) ----------------

def test_request_cache_hit_avoids_second_http_call(memdb, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        return _response(url, json_body={"default_branch": "main"})

    monkeypatch.setattr(gh.httpx, "get", fake_get)
    url = "https://api.github.com/repos/octocat/Hello-World"
    gh._request(memdb, url, None, gh.CACHE_TTL_MINUTES)
    gh._request(memdb, url, None, gh.CACHE_TTL_MINUTES)
    assert calls["n"] == 1


def test_request_raises_on_404(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return _response(url, status_code=404, json_body={"message": "Not Found"})

    monkeypatch.setattr(gh.httpx, "get", fake_get)
    with pytest.raises(gh.GitHubRepoError):
        gh._request(None, "https://api.github.com/repos/nope/nope", None, gh.CACHE_TTL_MINUTES)


def test_request_raises_rate_limit_error_with_reset_time(monkeypatch):
    reset_ts = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())

    def fake_get(url, headers=None, timeout=None):
        return _response(url, status_code=403, json_body={"message": "rate limited"},
                          headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset_ts)})

    monkeypatch.setattr(gh.httpx, "get", fake_get)
    with pytest.raises(gh.GitHubRateLimitError) as exc_info:
        gh._request(None, "https://api.github.com/repos/x/y", None, gh.CACHE_TTL_MINUTES)
    assert exc_info.value.reset_at == datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------- compute_github_dora orchestration ----------------

def test_compute_github_dora_no_releases_no_label(monkeypatch):
    monkeypatch.setattr(gh, "fetch_default_branch", lambda con, o, r, token=None: "main")
    monkeypatch.setattr(gh, "fetch_releases", lambda con, o, r, token=None: [])

    result = gh.compute_github_dora("octocat", "empty-repo", incident_label=None)
    assert result.deployment_frequency is None
    assert result.lead_time_for_changes is None
    assert result.change_failure_rate is None
    assert result.time_to_restore is None
    assert any("No releases found" in n for n in result.notes)
    assert any("No incident label provided" in n for n in result.notes)


def test_compute_github_dora_with_releases_and_label(monkeypatch):
    releases = [_release("2026-01-01T00:00:00Z"), _release("2026-02-01T00:00:00Z")]
    commits = [_commit("2026-01-15T00:00:00Z", "Add feature")]
    issues = [{"state": "closed", "created_at": "2026-01-01T00:00:00Z", "closed_at": "2026-01-01T03:00:00Z"}]

    monkeypatch.setattr(gh, "fetch_default_branch", lambda con, o, r, token=None: "main")
    monkeypatch.setattr(gh, "fetch_releases", lambda con, o, r, token=None: releases)
    monkeypatch.setattr(gh, "fetch_commits", lambda con, o, r, branch, since, token=None: commits)
    monkeypatch.setattr(gh, "fetch_issues", lambda con, o, r, label, token=None: issues)

    result = gh.compute_github_dora("octocat", "real-repo", incident_label="incident")
    assert result.deployment_frequency is not None
    assert result.change_failure_rate is not None
    assert result.lead_time_for_changes is not None
    assert result.time_to_restore is not None
