"""Day-2 detection matrix: stamping (CA-103), cross-feed (CA-107), resample (CA-108), range stats (CA-109)."""

from __future__ import annotations

import pandas as pd
import pytest

from chronoaudit import DeclaredConvention, Severity, align, audit, verify_resample
from fixtures import mutate_regime_shift, synth_fx

UTC_DECL = DeclaredConvention(tz="UTC", stamping="open", session_calendar="FX245")


@pytest.fixture(scope="module")
def clean():
    return synth_fx()


def _hits(findings, check_id, min_sev=Severity.WARN):
    return [f for f in findings if f.id == check_id and f.severity >= min_sev]


# ---- CA-103 stamping ----

def test_stamping_clean_inferred_open(clean):
    report = audit(clean, declared=UTC_DECL)
    assert report.inferred.stamping == "open"
    assert not _hits(report.findings, "CA-103"), "clean open-stamped data must not warn"


def test_stamping_close_mislabel_detected(clean):
    shifted = clean.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=1)  # relabel: now close-stamped
    report = audit(shifted, declared=UTC_DECL)  # declared open → contradiction
    hits = _hits(report.findings, "CA-103")
    assert hits, "CA-103 must catch close-stamped bars declared as open-stamped"
    assert report.inferred.stamping == "close"


# ---- CA-107 cross-feed ----

def test_align_clean_feeds(clean):
    report = align(clean, clean.copy(), UTC_DECL, UTC_DECL, max_lag_bars=15, window_days=30)
    assert report.verdict == "PASS"
    assert any("lag 0" in f.summary for f in report.findings)


def test_align_constant_lag_detected(clean):
    b = clean.copy()
    b.index = b.index + pd.Timedelta(minutes=7)
    report = align(clean, b, UTC_DECL, UTC_DECL, max_lag_bars=15, window_days=30)
    hits = _hits(report.findings, "CA-107", Severity.FATAL)
    assert hits, "CA-107 must catch a constant 7-minute lag"
    assert any(f.evidence.get("lag_bars") == 7 for f in hits)


def test_align_dst_regime_flip_detected(clean):
    b = mutate_regime_shift(clean, hours=1)
    report = align(clean, b, UTC_DECL, UTC_DECL, max_lag_bars=75, window_days=90)
    hits = _hits(report.findings, "CA-107", Severity.FATAL)
    assert any("flips between DST regimes" in f.summary for f in hits), \
        "CA-107 must diagnose the regime-dependent lag"


# ---- CA-108 resample causality ----

def _coarse(df, convention):
    # True constructions from open-stamped fine bars: an open-stamped coarse
    # bar labeled T covers [T, T+I); a close-stamped one labeled T covers
    # [T-I, T). Both bucket closed="left" — only the label side differs.
    label = "left" if convention == "open" else "right"
    return (
        df.resample("15min", label=label, closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def test_resample_consistent(clean):
    findings = verify_resample(clean, _coarse(clean, "open"), DeclaredConvention(stamping="open"))
    assert not [f for f in findings if f.severity >= Severity.WARN]


def test_resample_mislabeled_detected(clean):
    findings = verify_resample(clean, _coarse(clean, "close"), DeclaredConvention(stamping="open"))
    hits = [f for f in findings if f.id == "CA-108" and f.severity == Severity.FATAL]
    assert hits, "CA-108 must catch coarse bars built under the opposite stamping convention"


# ---- CA-111 session regime change ----

def test_session_regime_change_detected():
    from fixtures import mutate_short_session, synth_fx as _synth

    long_clean = _synth(weeks=20)  # ~4.6 months — enough monthly resolution
    mutated = mutate_short_session(long_clean, until="2024-04-01")
    report = audit(mutated, declared=UTC_DECL, checks=["CA-111"])
    hits = _hits(report.findings, "CA-111")
    assert hits, "CA-111 must catch a sustained session-length regime change"
    assert report.findings[-1].evidence["shifts"][0]["bars_per_day_after"] > \
        report.findings[-1].evidence["shifts"][0]["bars_per_day_before"]


def test_session_regime_stable_on_clean():
    from fixtures import synth_fx as _synth

    report = audit(_synth(weeks=20), declared=UTC_DECL, checks=["CA-111"])
    assert not _hits(report.findings, "CA-111"), "stable session must not warn"


# ---- CA-112 cross-instrument corroboration ----

def test_corroborate_instrument_specific_loss(clean):
    from chronoaudit import corroborate
    from fixtures import mutate_drop_range

    primary = mutate_drop_range(clean, "2024-02-13 14:00", "2024-02-13 15:00")
    report = corroborate(primary, clean, UTC_DECL, UTC_DECL)
    hits = _hits(report.findings, "CA-112")
    assert hits and hits[0].evidence["instrument_specific_losses"] == 1
    assert report.artifacts["gap_windows"][0]["class"] == "instrument_specific_loss"


def test_corroborate_market_wide_dark(clean):
    from chronoaudit import corroborate
    from fixtures import mutate_drop_range

    primary = mutate_drop_range(clean, "2024-02-13 14:00", "2024-02-13 15:00")
    reference = mutate_drop_range(clean, "2024-02-13 13:30", "2024-02-13 15:30")  # dark there too
    report = corroborate(primary, reference, UTC_DECL, UTC_DECL)
    ca112 = [f for f in report.findings if f.id == "CA-112"][0]
    assert ca112.severity == Severity.INFO
    assert report.artifacts["gap_windows"][0]["class"] == "market_wide_dark"


def test_corroborate_with_coarser_reference(clean):
    # Regression: expected-bar count must scale with the REFERENCE interval —
    # a clean 15m reference over a real 60min hole is 4 bars, not 60.
    from chronoaudit import corroborate
    from fixtures import mutate_drop_range

    primary = mutate_drop_range(clean, "2024-02-13 14:00", "2024-02-13 15:00")
    ref_15m = (clean.resample("15min")
               .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
               .dropna())
    report = corroborate(primary, ref_15m, UTC_DECL, UTC_DECL)
    assert report.artifacts["gap_windows"][0]["class"] == "instrument_specific_loss", \
        "coarse reference trading normally must still prove the loss is instrument-specific"


# ---- CME245 calendar ----

CME_DECL = DeclaredConvention(tz="UTC", stamping="open", session_calendar="CME245")


def test_cme_clean_break_not_flagged():
    from fixtures import synth_cme

    report = audit(synth_cme(), declared=CME_DECL, checks=["CA-104"])
    assert not _hits(report.findings, "CA-104"), \
        "daily maintenance break must net out to zero missing minutes"


def test_cme_dropped_hour_detected():
    from fixtures import mutate_drop_range, synth_cme

    mutated = mutate_drop_range(synth_cme(), "2024-02-13 14:00", "2024-02-13 15:00")
    report = audit(mutated, declared=CME_DECL, checks=["CA-104"])
    hits = _hits(report.findings, "CA-104")
    assert hits, "a real 60-minute hole must still be caught under CME245"


def test_cme_break_phantom_bars_detected():
    import numpy as np
    import pandas as pd
    from fixtures import synth_cme

    clean_cme = synth_cme()
    # Inject bars inside the Tuesday daily break: 17:30 ET = 22:30 UTC (winter).
    ts = pd.date_range("2024-02-13 22:30", periods=40, freq="1min")
    ref = clean_cme.iloc[0]
    phantom = pd.DataFrame(
        {c: np.full(len(ts), ref[c]) for c in ("open", "high", "low", "close")} | {"volume": 5.0},
        index=ts,
    )
    mutated = pd.concat([clean_cme, phantom]).sort_index()
    report = audit(mutated, declared=CME_DECL, checks=["CA-104"])
    hits = _hits(report.findings, "CA-104")
    assert hits and any("daily break" in f.summary for f in hits)


# ---- sparse-cache guard ----

def test_sparse_cache_skips_session_checks(clean):
    sparse = clean.iloc[::200]  # ~7 bars/day — a partial download, not a market
    report = audit(sparse, declared=UTC_DECL)
    hits = [f for f in report.findings if "SPARSE intraday coverage" in f.summary]
    assert hits, "sparse data must be flagged"
    assert not [f for f in report.findings if f.id in ("CA-101", "CA-102", "CA-103", "CA-104", "CA-111")], \
        "session-structure checks must be skipped on sparse data"


# ---- gaps manifest ----

def test_gaps_manifest_from_dropped_hour(clean):
    from fixtures import mutate_drop_range

    mutated = mutate_drop_range(clean, "2024-02-13 14:00", "2024-02-13 15:00")
    report = audit(mutated, declared=UTC_DECL)
    manifest = report.gaps_manifest(instrument="SYNTH")
    assert len(manifest["gaps"]) == 1
    gap = manifest["gaps"][0]
    assert gap["minutes"] == 60 and gap["start"].startswith("2024-02-13 14:00")
    assert not gap["holiday"]


# ---- CA-109 range stats ----

def test_range_stats_emitted(clean):
    report = audit(clean, declared=UTC_DECL)
    stats = [f for f in report.findings if f.id == "CA-109"]
    assert stats and "overall_median" in stats[0].evidence
    assert len(stats[0].evidence["hourly_median"]) == 24
