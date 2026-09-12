"""CA-107 (cross-feed alignment) and CA-108 (resample causality).

CA-107: two feeds of the same instrument, aligned to a common UTC grid;
lag estimated by shifting one close-return series against the other.
The killer diagnostic is the per-DST-regime lag: a feed pair where the
lag differs by 60 minutes between regimes means exactly one of them is
mislabeled or non-shifting — the classic silent corruptor of any
cross-feed strategy port (futures → CFD).

CA-108: rebuild a coarse dataset from its fine source under the declared
stamping convention and diff; if the OPPOSITE convention matches better,
every consumer of the coarse bars sees a one-bar shift, which is look-ahead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chronoaudit.core.models import AuditReport, DeclaredConvention, Finding, InferredConvention, Severity
from chronoaudit.infer.dst import us_dst_mask
from chronoaudit.io.loaders import infer_interval, load_ohlcv

MIN_CORR = 0.5
MIN_SAMPLES = 100  # fewer common bars than this cannot support a lag verdict


def _to_utc(df: pd.DataFrame, declared: DeclaredConvention) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out.index = idx.tz_localize(declared.tz, nonexistent="NaT", ambiguous="NaT").tz_convert("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = out[~out.index.isna()]
    return out


def _best_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag k maximizing corr(x[t], y[t+k]); positive k = y lags x.

    Returns (0, nan) when no lag had enough overlapping samples — callers
    must treat a non-finite correlation as INCONCLUSIVE, never as a verdict.
    """
    x = np.nan_to_num(x - np.nanmean(x))
    y = np.nan_to_num(y - np.nanmean(y))
    best_k, best_c = 0, -np.inf
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            a, b = x[: max(len(x) - k, 0)], y[k:]
        else:
            a, b = x[-k:], y[: max(len(y) + k, 0)]
        if min(len(a), len(b)) < MIN_SAMPLES:
            continue
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        c = float((a * b).sum() / denom) if denom > 0 else 0.0
        if c > best_c:
            best_k, best_c = k, c
    if not np.isfinite(best_c):
        return 0, float("nan")
    return best_k, best_c


def align(
    source_a: pd.DataFrame | str | Path,
    source_b: pd.DataFrame | str | Path,
    declared_a: DeclaredConvention | None = None,
    declared_b: DeclaredConvention | None = None,
    max_lag_bars: int = 180,
    window_days: int = 90,
) -> AuditReport:
    """CA-107: quantify timestamp alignment between two feeds of one instrument."""
    declared_a = declared_a or DeclaredConvention()
    declared_b = declared_b or DeclaredConvention()
    a = _to_utc(load_ohlcv(source_a), declared_a)
    b = _to_utc(load_ohlcv(source_b), declared_b)

    interval = max(infer_interval(a), infer_interval(b))
    grid_a = a["close"].resample(interval).last()
    grid_b = b["close"].resample(interval).last()
    start = max(grid_a.index[0], grid_b.index[0])
    end = min(grid_a.index[-1], grid_b.index[-1])
    findings: list[Finding] = []
    if start >= end:
        findings.append(Finding(id="CA-107", severity=Severity.WARN, summary="feeds do not overlap in time"))
        return _align_report(a, b, declared_a, findings, interval)

    joined = pd.concat(
        [grid_a[start:end], grid_b[start:end]], axis=1, keys=["a", "b"]
    ).astype(float)
    # Non-positive prices (vendor placeholders for empty bars) would become
    # ±inf under log and wipe out every lag correlation — mask them.
    joined = joined.where(joined > 0)
    rets = np.log(joined).diff()

    # Base-lag window: the last `window_days` days, but never fewer than
    # 500 grid rows when the overlap has them — a 90-day window on DAILY
    # bars is only ~65 rows, below any defensible verdict threshold.
    tail_by_days = rets[rets.index >= rets.index[-1] - pd.Timedelta(days=window_days)]
    recent = tail_by_days if len(tail_by_days) >= 500 else rets.tail(500)
    eff_max_lag = min(max_lag_bars, max(len(recent) // 3, 1))
    lag, corr = _best_lag(recent["a"].to_numpy(), recent["b"].to_numpy(), eff_max_lag)
    lag_td = lag * interval
    evid = {
        "lag_bars": lag,
        "lag_time": str(lag_td),
        "corr_at_best_lag": round(corr, 4) if np.isfinite(corr) else None,
        "grid_interval": str(interval),
        "window_rows": int(len(recent)),
        "max_lag_bars_used": int(eff_max_lag),
        "overlap": [str(start), str(end)],
    }

    if not np.isfinite(corr):
        findings.append(Finding(
            id="CA-107", severity=Severity.WARN,
            summary=(
                f"insufficient overlapping data to measure alignment "
                f"(need ≥{MIN_SAMPLES} common bars at grid {interval}; usable rows: {len(recent)})"
            ),
            evidence=evid,
            remediation="provide a longer overlap or finer bars; no alignment verdict was made",
        ))
        return _align_report(a, b, declared_a, findings, interval)

    stamping_differs = declared_a.stamping != declared_b.stamping
    if corr < MIN_CORR:
        findings.append(Finding(
            id="CA-107", severity=Severity.WARN,
            summary=f"feeds correlate weakly even at best lag (corr {corr:.3f}) — same instrument?",
            evidence=evid,
        ))
    elif lag != 0 and stamping_differs and abs(lag) == 1:
        findings.append(Finding(
            id="CA-107", severity=Severity.INFO,
            summary=(
                f"one-bar lag ({lag_td}, corr {corr:.3f}) is consistent with the differing declared "
                f"stamping conventions (A={declared_a.stamping}, B={declared_b.stamping}) — "
                "not corruption if both declarations are correct"
            ),
            evidence=evid,
            remediation="normalize both feeds to one stamping convention before joint use",
        ))
    elif lag != 0:
        findings.append(Finding(
            id="CA-107", severity=Severity.FATAL,
            summary=f"feeds misaligned: B lags A by {lag_td} (corr {corr:.3f})",
            evidence=evid,
            remediation="shift one feed by the measured lag or fix its declared tz before any cross-feed use",
        ))
    else:
        findings.append(Finding(id="CA-107", severity=Severity.INFO,
                                summary=f"feeds aligned at lag 0 (corr {corr:.3f})", evidence=evid))

    # Per-DST-regime lag — the mislabeled/non-shifting-feed detector.
    naive_idx = rets.index.tz_localize(None)
    dst_m = us_dst_mask(naive_idx)
    if dst_m.any() and (~dst_m).any():
        lag_std, c_std = _best_lag(rets["a"].to_numpy()[~dst_m], rets["b"].to_numpy()[~dst_m], eff_max_lag)
        lag_dst, c_dst = _best_lag(rets["a"].to_numpy()[dst_m], rets["b"].to_numpy()[dst_m], eff_max_lag)
        conclusive = np.isfinite(c_std) and np.isfinite(c_dst) and min(c_std, c_dst) >= MIN_CORR
        regime_evid = {
            "lag_standard_regime": str(lag_std * interval),
            "lag_dst_regime": str(lag_dst * interval),
            "corr_standard": round(c_std, 3) if np.isfinite(c_std) else None,
            "corr_dst": round(c_dst, 3) if np.isfinite(c_dst) else None,
            "rows_standard": int((~dst_m).sum()),
            "rows_dst": int(dst_m.sum()),
        }
        if conclusive and lag_std != lag_dst:
            findings.append(Finding(
                id="CA-107", severity=Severity.FATAL,
                summary=(
                    f"lag flips between DST regimes ({lag_std * interval} vs {lag_dst * interval}) — "
                    "one feed is DST-mislabeled or non-shifting; every time-of-day comparison across these feeds is corrupt"
                ),
                evidence=regime_evid,
                remediation="re-run chronoaudit CA-101 on each feed separately to identify which one lies",
            ))
        elif conclusive:
            findings.append(Finding(id="CA-107", severity=Severity.INFO,
                                    summary="lag stable across DST regimes", evidence=regime_evid))
        else:
            findings.append(Finding(
                id="CA-107", severity=Severity.INFO,
                summary="insufficient data to compare lags across DST regimes — no cross-regime verdict",
                evidence=regime_evid,
            ))
    return _align_report(a, b, declared_a, findings, interval)


def _align_report(a, b, declared_a, findings, interval) -> AuditReport:
    return AuditReport(
        dataset={
            "mode": "align",
            "rows_a": int(len(a)), "rows_b": int(len(b)),
            "interval": str(interval),
        },
        declared=declared_a,
        inferred=InferredConvention(),
        findings=findings,
    )


def verify_resample(
    fine: pd.DataFrame | str | Path,
    coarse: pd.DataFrame | str | Path,
    declared_coarse: DeclaredConvention,
    fine_stamping: str = "open",
    rel_tol: float = 1e-9,
) -> list[Finding]:
    """CA-108: does the coarse dataset follow its declared stamping convention?

    fine_stamping: the FINE feed's stamping convention ("open"/"close").
    The rebuild must know it — bucketing close-stamped minutes as if they
    were open-stamped shifts every coarse bar by one fine bar, which both
    breaks the match on honest data and hides a real one-bar look-ahead.
    """
    f = load_ohlcv(fine)
    c = load_ohlcv(coarse)
    interval = infer_interval(c)

    # Normalize the fine feed to open-stamping internally so one bucketing
    # rule serves both cases: a close-stamped fine bar labeled t covers
    # (t - fine_interval, t] — relabel it to its open time.
    if fine_stamping == "close":
        f = f.copy()
        f.index = f.index - infer_interval(f)

    # Bucket boundaries must sit on the COARSE feed's own grid, not on
    # midnight: FX/CME daily bars anchor at 17:00 ET, MT5 brokers stamp 4h
    # bars on 01:00/05:00 grids. Origin only matters modulo the interval,
    # so the first coarse label anchors both conventions.
    origin = c.index[0]

    def rebuild(convention: str) -> pd.DataFrame:
        # open-stamped coarse bar labeled T covers [T, T+I);
        # close-stamped coarse bar labeled T covers [T-I, T).
        label = "left" if convention == "open" else "right"
        agg = f.resample(interval, closed="left", label=label, origin=origin).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        )
        return agg.dropna()

    def match_frac(rebuilt: pd.DataFrame) -> float:
        j = rebuilt.join(c, how="inner", lsuffix="_r", rsuffix="_c")
        if j.empty:
            return 0.0
        ok = np.ones(len(j), dtype=bool)
        for col in ("open", "high", "low", "close"):
            ok &= np.isclose(j[f"{col}_r"], j[f"{col}_c"], rtol=rel_tol, equal_nan=False)
        return float(ok.mean())

    other = "close" if declared_coarse.stamping == "open" else "open"
    frac_declared = match_frac(rebuild(declared_coarse.stamping))
    frac_other = match_frac(rebuild(other))
    evid = {
        "match_declared_convention": round(frac_declared, 4),
        "match_opposite_convention": round(frac_other, 4),
        "coarse_interval": str(interval),
        "fine_stamping_assumed": fine_stamping,
    }

    if frac_other > frac_declared + 0.10:
        return [Finding(
            id="CA-108", severity=Severity.FATAL,
            summary=(
                f"coarse bars match the OPPOSITE stamping convention "
                f"({frac_other:.1%} vs {frac_declared:.1%}) — declared {declared_coarse.stamping}-stamped "
                "but built the other way; one-bar look-ahead for every consumer"
            ),
            evidence=evid,
            remediation="fix the declaration or re-shift the coarse index by one interval",
        )]
    if max(frac_declared, frac_other) < 0.5:
        return [Finding(
            id="CA-108", severity=Severity.WARN,
            summary=f"coarse bars match fine source poorly under both conventions (best {max(frac_declared, frac_other):.1%}) — different source or heavy filtering",
            evidence=evid,
        )]
    return [Finding(id="CA-108", severity=Severity.INFO,
                    summary=f"coarse bars consistent with declared {declared_coarse.stamping}-stamping ({frac_declared:.1%} match)",
                    evidence=evid)]
