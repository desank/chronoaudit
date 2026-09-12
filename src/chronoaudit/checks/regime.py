"""CA-111: session-length regime-change detector.

Found in the wild (first production run): an index CFD whose session was
US-hours-only (~835 bars/day) for its first three months, then extended to
~22h. Every backtest spanning the boundary silently mixes two structurally
different markets — overnight signals simply don't exist in the early
regime. This check detects sustained level shifts in bars-per-day.
"""

from __future__ import annotations

import pandas as pd

from chronoaudit.checks.base import AuditContext, register
from chronoaudit.core.models import Finding, Severity

MIN_MONTHS = 4
SHIFT_FRAC = 0.15      # month-over-month median shift that counts as a break
SUSTAIN_FRAC = 0.10    # next month must stay within this of the new level


@register("CA-111")
def session_regime_change(df: pd.DataFrame, ctx: AuditContext) -> list[Finding]:
    counts = pd.Series(1, index=df.index).groupby(df.index.date).sum()
    weekday = pd.Series([pd.Timestamp(d).dayofweek for d in counts.index], index=counts.index)
    wk = counts[weekday < 5]
    if wk.empty:
        return []
    monthly = wk.groupby(pd.PeriodIndex(pd.DatetimeIndex(wk.index), freq="M")).median()
    if len(monthly) < MIN_MONTHS:
        return []

    shifts = []
    vals = monthly.to_numpy(dtype=float)
    for i in range(1, len(vals) - 1):
        prev, cur, nxt = vals[i - 1], vals[i], vals[i + 1]
        if prev <= 0:
            continue
        if abs(cur - prev) / prev > SHIFT_FRAC and abs(nxt - cur) / max(cur, 1.0) < SUSTAIN_FRAC:
            shifts.append((str(monthly.index[i]), float(prev), float(cur)))

    if not shifts:
        return [Finding(id="CA-111", severity=Severity.INFO,
                        summary="session length stable across the dataset",
                        evidence={"monthly_median_bars_per_day_range":
                                  [float(vals.min()), float(vals.max())]})]

    return [Finding(
        id="CA-111",
        severity=Severity.WARN,
        summary=(
            f"{len(shifts)} sustained session-length regime change(s) — "
            "backtests spanning these boundaries mix structurally different markets"
        ),
        evidence={"shifts": [
            {"month": m, "bars_per_day_before": b, "bars_per_day_after": a} for m, b, a in shifts
        ]},
        affected=[m for m, _, _ in shifts],
        remediation="window backtests to a single regime, or model the session change explicitly",
    )]
