"""Detection matrix: every mutation caught, clean data silent.

This is the acceptance bar from the design spec — we test the DETECTOR,
not just run it (same ethos as backtest sim-fidelity: an audit tool that
misses injected corruption would itself be the artifact factory).
"""

from __future__ import annotations

import json

import pytest

from chronoaudit import DeclaredConvention, Severity, audit
from fixtures import (
    mutate_drop_range,
    mutate_duplicates,
    mutate_ohlc_break,
    mutate_regime_shift,
    mutate_stale,
    mutate_weekend_bars,
    synth_fx,
)

UTC_DECL = DeclaredConvention(tz="UTC", stamping="open", session_calendar="FX245")


@pytest.fixture(scope="module")
def clean():
    return synth_fx()


def _hits(report, check_id, min_sev=Severity.WARN):
    return [f for f in report.findings if f.id == check_id and f.severity >= min_sev]


def test_clean_data_passes(clean):
    report = audit(clean, declared=UTC_DECL)
    bad = [f for f in report.findings if f.severity > Severity.INFO]
    assert report.verdict == "PASS", f"clean data flagged: {[f.summary for f in bad]}"


def test_declared_tz_mislabel_detected(clean):
    # Data is genuinely UTC; user declares market-local ET. Classic vendor-docs error.
    report = audit(clean, declared=DeclaredConvention(tz="America/New_York", session_calendar="FX245"))
    assert _hits(report, "CA-101", Severity.FATAL), "CA-101 must catch declared-tz mislabel"
    assert report.verdict == "FAIL"


def test_broker_dst_regime_shift_detected(clean):
    # Summer rows shifted +1h: raw wall times constant year-round, but declared UTC.
    mutated = mutate_regime_shift(clean, hours=1)
    report = audit(mutated, declared=UTC_DECL)
    assert _hits(report, "CA-101", Severity.FATAL), "CA-101 must catch DST-regime shift corruption"


def test_dropped_hour_detected(clean):
    mutated = mutate_drop_range(clean, "2024-02-13 14:00", "2024-02-13 15:00")
    report = audit(mutated, declared=UTC_DECL)
    assert _hits(report, "CA-104"), "CA-104 must catch a 60-bar data hole"


def test_phantom_weekend_bars_detected(clean):
    mutated = mutate_weekend_bars(clean, "2024-02-17 12:00")  # Saturday noon UTC
    report = audit(mutated, declared=UTC_DECL)
    hits = _hits(report, "CA-104")
    assert hits and any("weekend" in f.summary for f in hits), "CA-104 must catch weekend phantoms"


def test_stale_plateau_detected(clean):
    mutated = mutate_stale(clean, "2024-02-14 09:00", n=45)
    report = audit(mutated, declared=UTC_DECL)
    hits = _hits(report, "CA-106")
    assert hits and any("plateau" in f.summary for f in hits), "CA-106 must catch feed freeze"


def test_ohlc_violation_detected(clean):
    mutated = mutate_ohlc_break(clean, "2024-02-14 10:00")
    report = audit(mutated, declared=UTC_DECL)
    assert _hits(report, "CA-106"), "CA-106 must catch low>high bars"


def test_duplicate_timestamps_detected(clean):
    mutated = mutate_duplicates(clean, "2024-02-14 11:00")
    report = audit(mutated, declared=UTC_DECL)
    assert _hits(report, "CA-105"), "CA-105 must catch duplicate timestamps"


def test_duplicates_at_week_open_edge_do_not_crash(clean):
    # Regression: get_loc-based gap lookup returned a slice (crash) when the
    # duplicated timestamp was itself a weekend-gap edge.
    mutated = mutate_duplicates(clean, "2024-02-11 22:00", n=3)  # first bars after weekend (UTC)
    report = audit(mutated, declared=UTC_DECL)
    assert _hits(report, "CA-105"), "duplicates at the gap edge must be reported, not crash the audit"


def test_report_serialization(clean):
    report = audit(clean, declared=UTC_DECL)
    payload = json.loads(report.to_json())
    assert payload["verdict"] == "PASS"
    assert payload["declared"]["tz"] == "UTC"
    assert isinstance(payload["findings"], list)
    assert "chronoaudit report" in report.to_markdown()
