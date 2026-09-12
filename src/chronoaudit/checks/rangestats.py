"""CA-109: finer-bar range statistics (stop-vs-intrabar-noise reference).

Report-only: per-hour median and quartiles of bar range. The consuming
rule: if a strategy's median stop distance ≲ 3× the next-finer timeframe's
median bar range, re-simulate at finer resolution before trusting results.
"""

from __future__ import annotations

import pandas as pd

from chronoaudit.checks.base import AuditContext, register
from chronoaudit.core.models import Finding, Severity


@register("CA-109")
def finer_bar_range_stats(df: pd.DataFrame, ctx: AuditContext) -> list[Finding]:
    rng = df["high"] - df["low"]
    by_hour = rng.groupby(df.index.hour)
    stats = {
        "overall_median": float(rng.median()),
        "overall_p25": float(rng.quantile(0.25)),
        "overall_p75": float(rng.quantile(0.75)),
        "hourly_median": {int(h): round(float(v), 8) for h, v in by_hour.median().items()},
    }
    return [Finding(
        id="CA-109",
        severity=Severity.INFO,
        summary=f"bar-range stats at {ctx.interval} (sizing reference: a stop ≲ 3× the finer-bar median range needs re-simulating on finer bars)",
        evidence=stats,
    )]
