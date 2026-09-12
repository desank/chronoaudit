"""Detection-matrix additions from the 2026-07-05 pre-launch hardening pass.

Each test pins one public-data failure class found by the hostile-input sweep
or the adversarial review: daily bars, unknown calendars, epoch time columns,
MT5 exports, non-numeric prices, CLI exit codes, daily cross-feed alignment,
close-stamped resample verification, and non-US-DST markets (B11).
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chronoaudit
from chronoaudit import DeclaredConvention, align, audit, load_ohlcv, verify_resample
from chronoaudit.cli import main as cli_main

rng = np.random.default_rng(11)


def _ohlcv(index: pd.DatetimeIndex, scale: float = 1e-4, base: float = 1.1) -> pd.DataFrame:
    """Coherent OHLCV: high/low always bracket open/close."""
    n = len(index)
    px = base + np.cumsum(rng.normal(0, scale, n))
    close = px + rng.normal(0, scale / 3, n)
    wick = np.abs(rng.normal(0, scale / 2, n))
    return pd.DataFrame(
        {
            "open": px,
            "high": np.maximum(px, close) + wick,
            "low": np.minimum(px, close) - wick,
            "close": close,
            "volume": rng.integers(10, 500, n).astype(float),
        },
        index=index,
    )


def _daily(days: int = 780, start: str = "2022-01-03") -> pd.DataFrame:
    return _ohlcv(pd.bdate_range(start, periods=days), scale=1.0, base=100.0)


def _fx_1m(days: int = 10, start: str = "2024-03-04") -> pd.DataFrame:
    idx = []
    d0 = pd.Timestamp(start)
    for d in range(days):
        day = d0 + pd.Timedelta(days=d)
        if day.dayofweek < 5:
            idx.append(pd.date_range(day, day + pd.Timedelta(hours=22), freq="1min", inclusive="left"))
    return _ohlcv(idx[0].append(idx[1:]))


def _finding_ids(report, sev=None):
    return [f.id for f in report.findings if sev is None or f.severity.name == sev]


# ---------- B3/S1: coarse and daily data must not get the sparse scare ----------

def test_daily_bars_get_calm_coarse_note_not_sparse_warn():
    r = audit(_daily(), declared=DeclaredConvention(tz="UTC"))
    ca113 = [f for f in r.findings if f.id == "CA-113"]
    assert len(ca113) == 1 and ca113[0].severity.name == "INFO"
    assert "too coarse" in ca113[0].summary
    # session-structure checks must not have run
    assert not {"CA-101", "CA-102", "CA-103", "CA-104", "CA-111"} & set(_finding_ids(r))
    # and no vendor-specific vocabulary anywhere
    for f in r.findings:
        assert "spread cap" not in (f.remediation + f.summary)


def test_sparse_intraday_cache_still_warns_generically():
    idx = pd.date_range("2024-03-04", periods=40, freq="1D") + pd.Timedelta(hours=3)
    thin = _ohlcv(pd.DatetimeIndex(np.repeat(idx.values, 3)) + pd.to_timedelta([0, 1, 2] * len(idx), unit="min"))
    r = audit(thin, declared=DeclaredConvention(tz="UTC", bar_interval=pd.Timedelta(minutes=1)))
    ca113 = [f for f in r.findings if f.id == "CA-113" and f.severity.name == "WARN"]
    assert len(ca113) == 1 and "SPARSE" in ca113[0].summary
    assert "contract series" in ca113[0].remediation  # generic wording, no vendor-specific context


# ---------- B2: unknown calendar must be loud ----------

def test_unknown_calendar_emits_partial_audit_warning():
    r = audit(_fx_1m(), declared=DeclaredConvention(tz="UTC", session_calendar="XNYS"))
    ca110 = [f for f in r.findings if f.id == "CA-110"]
    assert len(ca110) == 1 and ca110[0].severity.name == "WARN"
    assert "PARTIAL" in ca110[0].summary and "XNYS" in ca110[0].summary


# ---------- B5/B6: loader hardening ----------

def test_epoch_seconds_and_ms_time_columns(tmp_path):
    df = _fx_1m(days=3).reset_index().rename(columns={"index": "ts"})
    true_start = df["ts"].iloc[0]
    # as_unit("ns") first: pandas 3.x may build the index in us/ms resolution,
    # and a raw int64 view in the wrong unit yields wrong epoch values —
    # the exact bug class this library audits for.
    epoch_ns = pd.DatetimeIndex(df["ts"]).as_unit("ns").asi8
    for unit, factor in (("s", 10**9), ("ms", 10**6)):
        out = df.copy()
        out["time"] = epoch_ns // factor
        p = tmp_path / f"epoch_{unit}.csv"
        out.drop(columns=["ts"]).to_csv(p, index=False)
        loaded = load_ohlcv(str(p))
        assert loaded.index[0] == true_start, unit


def test_mt5_terminal_export_brackets_and_split_datetime(tmp_path):
    df = _fx_1m(days=3).reset_index().rename(columns={"index": "ts"})
    out = pd.DataFrame({
        "<DATE>": df["ts"].dt.strftime("%Y.%m.%d"),
        "<TIME>": df["ts"].dt.strftime("%H:%M:%S"),
        "<OPEN>": df["open"], "<HIGH>": df["high"], "<LOW>": df["low"],
        "<CLOSE>": df["close"], "<TICKVOL>": df["volume"],
    })
    p = tmp_path / "mt5.csv"
    out.to_csv(p, sep="\t", index=False)
    loaded = load_ohlcv(str(p))
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert loaded.index[0] == df["ts"].iloc[0]


def test_non_numeric_prices_fail_loud_at_load(tmp_path):
    df = _fx_1m(days=2).reset_index().rename(columns={"index": "ts"})
    df["open"] = df["open"].map(lambda v: f"{v:.5f}".replace(".", ","))
    p = tmp_path / "eu.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="not numeric"):
        load_ohlcv(str(p))


def test_row_number_time_column_rejected(tmp_path):
    df = _fx_1m(days=2).reset_index(drop=True)
    df["time"] = np.arange(len(df))
    p = tmp_path / "rownum.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="row numbers|epoch"):
        load_ohlcv(str(p))


# ---------- B1: CLI exit codes ----------

def test_cli_missing_file_exits_3_not_1(capsys):
    rc = cli_main(["audit", "definitely_not_here.csv"])
    assert rc == 3
    assert "error" in capsys.readouterr().err.lower()


def test_cli_version_flag():
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0


# ---------- B9/S10: cross-feed alignment on daily bars & dirty prices ----------

def test_align_identical_daily_feeds_reports_aligned():
    d = _daily()
    r = align(d, d.copy(), DeclaredConvention(tz="UTC"), DeclaredConvention(tz="UTC"))
    warns = [f for f in r.findings if f.severity.name == "WARN"]
    assert not any("same instrument" in f.summary for f in warns)
    assert any("aligned at lag 0" in f.summary for f in r.findings)


def test_align_shifted_daily_pair_is_fatal():
    d = _daily()
    shifted = d.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    r = align(d, shifted, DeclaredConvention(tz="UTC"), DeclaredConvention(tz="UTC"))
    assert any(f.severity.name == "FATAL" and "misaligned" in f.summary for f in r.findings)


def test_align_short_daily_overlap_says_insufficient_not_suspicious():
    d = _daily(days=60)  # ~83 calendar-grid rows < MIN_SAMPLES — genuinely unmeasurable
    r = align(d, d.copy(), DeclaredConvention(tz="UTC"), DeclaredConvention(tz="UTC"))
    assert any("insufficient overlapping data" in f.summary for f in r.findings)
    assert not any("same instrument" in f.summary for f in r.findings)


def test_align_zero_price_placeholder_does_not_wipe_verdict():
    a = _fx_1m(days=10)
    b = a.copy()
    a.iloc[500, a.columns.get_loc("close")] = 0.0
    r = align(a, b, DeclaredConvention(tz="UTC"), DeclaredConvention(tz="UTC"))
    assert any("aligned at lag 0" in f.summary for f in r.findings)


# ---------- B10/S11: resample causality, close-stamped and off-midnight ----------

def _true_close_stamped(fine: pd.DataFrame, interval: str) -> pd.DataFrame:
    # A close-stamped coarse bar labeled T covers open-stamped fine bars [T-I, T).
    return (
        fine.resample(interval, closed="left", label="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def test_verify_resample_truthful_close_declaration_passes():
    fine = _fx_1m(days=5)
    coarse = _true_close_stamped(fine, "1h")
    fs = verify_resample(fine, coarse, DeclaredConvention(stamping="close"))
    assert fs[0].severity.name == "INFO"
    assert fs[0].evidence["match_declared_convention"] > 0.99


def test_verify_resample_catches_the_one_bar_lookahead():
    fine = _fx_1m(days=5)
    coarse = _true_close_stamped(fine, "1h")  # truly close-stamped …
    fs = verify_resample(fine, coarse, DeclaredConvention(stamping="open"))  # … declared open
    assert fs[0].severity.name == "FATAL"
    assert fs[0].evidence["match_opposite_convention"] > 0.99


def test_verify_resample_off_midnight_grid_anchor():
    fine = _fx_1m(days=5)
    origin = fine.index[0] + pd.Timedelta(hours=1)  # 01:00-anchored 4h grid (MT5-style)
    coarse = (
        fine.resample("4h", closed="left", label="left", origin=origin)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .iloc[1:-1]
    )
    fs = verify_resample(fine, coarse, DeclaredConvention(stamping="open"))
    assert fs[0].severity.name == "INFO"
    assert fs[0].evidence["match_declared_convention"] > 0.99


# ---------- B11: no sweeping DST-family assumptions ----------

def _tokyo_like_year() -> pd.DataFrame:
    """A DST-free market: weekday sessions fixed at 00:00–06:00 UTC all year,
    stable hourly volume shape (peak at hour 2)."""
    idx = []
    for day in pd.bdate_range("2024-01-08", "2024-12-20"):
        idx.append(pd.date_range(day, day + pd.Timedelta(hours=6), freq="1min", inclusive="left"))
    ix = idx[0].append(idx[1:])
    df = _ohlcv(ix, scale=0.5, base=30000.0)
    hour = ix.hour
    df["volume"] = (100 + 80 * np.exp(-((hour - 2) ** 2))) * (1 + rng.normal(0, 0.02, len(ix)))
    return df


def test_unknown_family_market_gets_no_verdict_not_false_fatal():
    df = _tokyo_like_year()
    r = audit(df, declared=DeclaredConvention(tz="UTC", session_calendar="TSE"),
              checks=["CA-101"])
    fatals = [f for f in r.findings if f.severity.name == "FATAL"]
    assert not fatals, [f.summary for f in fatals]
    assert any("NO VERDICT" in f.summary for f in r.findings)


def test_declared_dst_free_market_passes_clean():
    df = _tokyo_like_year()
    r = audit(df, declared=DeclaredConvention(tz="UTC", session_calendar="TSE", session_dst="none"),
              checks=["CA-101"])
    assert not [f for f in r.findings if f.severity.name == "FATAL"]
    assert any("DST-free" in f.summary for f in r.findings)


def test_us_family_verdict_still_bites_when_family_is_declared():
    # Same DST-free data, but the user CONFIRMS the market follows US DST:
    # now shift 0 under a fixed-offset declaration is a real contradiction.
    df = _tokyo_like_year()
    r = audit(df, declared=DeclaredConvention(tz="UTC", session_calendar="TSE", session_dst="us"),
              checks=["CA-101"])
    assert any(f.severity.name == "FATAL" for f in r.findings)


# ---------- S8: version single-sourcing ----------

def test_pyproject_version_matches_runtime():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    version = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert version == chronoaudit.__version__


# ---------- CA-101 tie reporting: an unverifiable sample must not PASS ----------
# Found 2026-09-12 auditing a real January-only GBPUSD file. The candidate list
# is led by the DECLARED zone and `max()` returns the first maximum, so a sample
# where every candidate ties handed the user's own claim back as "inferred tz=X
# (conf 1.00)" and returned PASS. Declaring UTC and declaring Europe/London on
# the same winter file both "passed" — they are identical in January and differ
# by an hour from late March.

def _winter_only_fx(weeks: int = 8) -> pd.DataFrame:
    """24/5 FX bars entirely inside US standard time (no DST boundary spanned).

    Hourly bars keep the fixture small; the weekend-edge fingerprint only needs
    the session edges, and the activity-profile fingerprint cannot run at all
    on a single-regime sample — which is the point of this fixture.
    """
    parts = []
    for w in range(weeks):
        sun = pd.Timestamp("2024-01-07") + pd.Timedelta(weeks=w)
        parts.append(
            pd.date_range(sun + pd.Timedelta(hours=17),
                          sun + pd.Timedelta(days=5, hours=17),
                          freq="1h", inclusive="left", tz="America/New_York")
        )
    et = parts[0]
    for p in parts[1:]:
        et = et.append(p)
    return _ohlcv(et.tz_convert("UTC").tz_localize(None))


def test_winter_only_sample_does_not_confirm_the_declared_tz():
    df = _winter_only_fx()
    r = audit(df, declared=DeclaredConvention(tz="UTC", session_calendar="FX245"),
              checks=["CA-101"])
    assert r.verdict != "PASS", "an unverifiable timezone must not report a clean PASS"
    assert any(f.id == "CA-101" and f.severity.name == "WARN" and "NOT VERIFIED" in f.summary
               for f in r.findings), [f.summary for f in r.findings]


def test_winter_only_sample_reports_the_tie_instead_of_echoing_the_claim():
    df = _winter_only_fx()
    r = audit(df, declared=DeclaredConvention(tz="Europe/London", session_calendar="FX245"),
              checks=["CA-101"])
    # The fingerprint cannot separate these, and must say so rather than
    # returning the declaration at confidence 1.00.
    assert len(r.inferred.tz_tied) > 1
    assert r.inferred.tz_tie_mixes_dst is True
    assert "UTC" in r.inferred.tz_tied
    assert r.inferred.dst_regime == "undetermined"
    assert "NOT CONFIRMED" in r.to_markdown()


def test_dst_spanning_sample_still_confirms_and_stays_quiet():
    # The regression guard: fingerprint 2 resolves what fingerprint 1 cannot,
    # so a sample spanning a DST boundary must NOT pick up the new warning.
    from fixtures import synth_fx

    r = audit(synth_fx(), declared=DeclaredConvention(tz="UTC", session_calendar="FX245"),
              checks=["CA-101"])
    assert not any("NOT VERIFIED" in f.summary for f in r.findings)
    assert r.verdict == "PASS"
    assert r.inferred.dst_regime == "none"


def test_same_dst_behaviour_tie_is_not_alarmed_about():
    # UTC vs Etc/GMT-2 vs Etc/GMT-3 are provably indistinguishable from bar
    # structure and behave identically for session work — a documented limit,
    # not a defect. It must read as a tie, never as "NOT CONFIRMED".
    from fixtures import synth_fx

    r = audit(synth_fx(), declared=DeclaredConvention(tz="UTC", session_calendar="FX245"),
              checks=["CA-101"])
    if len(r.inferred.tz_tied) > 1 and not r.inferred.tz_tie_mixes_dst:
        assert "NOT CONFIRMED" not in r.to_markdown()
        assert "same DST behaviour" in r.to_markdown()


def test_wrong_tz_on_a_dst_spanning_sample_still_fatals():
    # The tie logic must not soften a genuine contradiction.
    from fixtures import synth_fx

    r = audit(synth_fx(), declared=DeclaredConvention(tz="America/New_York",
                                                      session_calendar="FX245"),
              checks=["CA-101"])
    assert any(f.id == "CA-101" and f.severity.name == "FATAL" for f in r.findings)
    assert r.verdict == "FAIL"
