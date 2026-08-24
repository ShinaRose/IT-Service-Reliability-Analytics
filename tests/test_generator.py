from relplatform.generator.postmortems import CATEGORIES
from relplatform.generator.roster import ENGINEERS
from relplatform.generator.simulate import simulate


def test_generates_all_tables(sim_result):
    assert len(sim_result.deployments) > 0
    assert len(sim_result.incidents) > 0
    assert len(sim_result.alerts) > 0
    assert len(sim_result.resource_metrics) > 0


def test_alert_storm_size_bounds(sim_result):
    from collections import Counter

    counts = Counter(a["incident_id"] for a in sim_result.alerts if a["incident_id"])
    for inc_id, n in counts.items():
        assert n >= 30, f"{inc_id} storm too small: {n}"


def test_incident_categories_valid(sim_result):
    for inc in sim_result.incidents:
        assert inc["root_cause_category"] in CATEGORIES
        assert inc["resolved_at"] >= inc["acknowledged_at"] >= inc["started_at"]
        assert len(inc["postmortem_text"]) > 200


def test_deterministic_with_seed():
    r1 = simulate(seed=99, months=2)
    r2 = simulate(seed=99, months=2)
    assert len(r1.incidents) == len(r2.incidents)
    assert [i["id"] for i in r1.incidents] == [i["id"] for i in r2.incidents]


def test_noise_alerts_exist(sim_result):
    noise = [a for a in sim_result.alerts if a["incident_id"] is None]
    assert len(noise) > 0


def test_roster_covers_window_with_no_gaps_or_overlaps(sim_result):
    shifts = sorted(sim_result.on_call_shifts, key=lambda s: s["shift_start"])
    assert len(shifts) > 0
    for prev, nxt in zip(shifts, shifts[1:]):
        assert prev["shift_end"] == nxt["shift_start"], "roster must have no gap or overlap between consecutive shifts"


def test_roster_engineers_are_from_the_known_pool(sim_result):
    for shift in sim_result.on_call_shifts:
        assert shift["engineer"] in ENGINEERS


def test_roster_swapped_only_on_holiday_shifts(sim_result):
    for shift in sim_result.on_call_shifts:
        if shift["swapped"]:
            assert shift["is_holiday"]


def test_roster_deterministic_with_seed():
    r1 = simulate(seed=99, months=2)
    r2 = simulate(seed=99, months=2)
    assert [s["engineer"] for s in r1.on_call_shifts] == [s["engineer"] for s in r2.on_call_shifts]
