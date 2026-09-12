"""CA-104: session/calendar conformance (FX245 and CME245 session specs).

Holes are judged NET of structural closures (weekend, daily maintenance
break) so a gap spanning the CME 17:00–18:00 ET break isn't 61 phantom
missing minutes. Holiday-window holes report at INFO (legitimate closures).
"""

from __future__ import annotations

import pandas as pd

from chronoaudit.calendars.sessions import (
    SPECS,
    SessionSpec,
    expected_closed_minutes,
    holiday_dates,
    in_structural_closure,
)
from chronoaudit.checks.base import AuditContext, register
from chronoaudit.core.models import Finding, Severity

MISSING_RUN_WARN = 30  # net missing time ≥ 30 bar-intervals is suspicious


@register("CA-104")
def session_calendar_conformance(df: pd.DataFrame, ctx: AuditContext) -> list[Finding]:
    spec = SPECS.get(ctx.declared.session_calendar)
    if spec is None:
        return []

    findings: list[Finding] = []
    tz = ctx.inferred.tz if ctx.inferred.tz_confidence >= 0.9 else ctx.declared.tz
    idx = df.index
    try:
        et = (idx.tz_localize(tz, nonexistent="NaT", ambiguous="NaT") if idx.tz is None else idx).tz_convert(
            "America/New_York"
        )
    except Exception as exc:
        findings.append(
            Finding(id="CA-104", severity=Severity.WARN,
                    summary=f"could not interpret timestamps in tz={tz}: {exc}")
        )
        return findings

    # Phantom bars: weekend-gap phantoms are a hard signal (weekends don't
    # move); daily-break phantoms cap at WARN — break windows shift across
    # vendor/exchange history (CME moved equity maintenance hours in 2015,
    # which otherwise reports tens of thousands of false 'phantoms' on any
    # dataset spanning that change).
    break_spec_removed = SessionSpec(spec.name, spec.week_close, spec.week_open, None, spec.us_market_holidays)
    weekend_ph = in_structural_closure(et, break_spec_removed)
    all_ph = in_structural_closure(et, spec)
    break_ph = all_ph & ~weekend_ph
    if weekend_ph.any():
        frac = weekend_ph.mean()
        findings.append(
            Finding(
                id="CA-104",
                severity=Severity.FATAL if frac > 0.001 else Severity.WARN,
                summary=(
                    f"{int(weekend_ph.sum())} bars inside the weekend gap / structural closures "
                    f"({frac:.4%}) under tz={tz}, calendar={spec.name}"
                ),
                evidence={"count": int(weekend_ph.sum()), "fraction": float(frac), "tz_used": tz},
                affected=[str(t) for t in idx[weekend_ph][:20]],
                remediation="phantom weekend bars poison session stats; drop them or fix the tz/calendar declaration",
            )
        )
    if break_ph.any():
        ph_years = sorted(set(et[break_ph].year))
        findings.append(
            Finding(
                id="CA-104",
                severity=Severity.WARN,
                summary=(
                    f"{int(break_ph.sum())} bars inside the declared daily break "
                    f"({break_ph.mean():.4%}) — possible historical session-hours change, "
                    f"not necessarily corrupt data (years: {ph_years[0]}–{ph_years[-1]})"
                ),
                evidence={"count": int(break_ph.sum()), "years": ph_years[:20], "tz_used": tz},
                affected=[str(t) for t in idx[break_ph][:10]],
                remediation="check exchange session-hours history for these years; consider era-specific specs",
            )
        )

    # Data holes: gross runs, then netted against expected closures.
    diffs = idx.to_series().diff().dropna()
    intraweek = diffs[(diffs > ctx.interval) & (diffs <= pd.Timedelta(hours=24))]
    gross_runs = intraweek[intraweek >= ctx.interval * MISSING_RUN_WARN]
    net_threshold_min = int(MISSING_RUN_WARN * (ctx.interval / pd.Timedelta(minutes=1)))

    gap_windows = []
    if not gross_runs.empty:
        holidays = holiday_dates(spec, range(idx[0].year, idx[-1].year + 1))
        et_by_pos = pd.Series(et, index=idx)
        for end_ts, run in gross_runs.items():
            start_ts = end_ts - run + ctx.interval
            end_et = et_by_pos.loc[end_ts]
            end_et = end_et.iloc[0] if isinstance(end_et, pd.Series) else end_et
            start_et = end_et - run + ctx.interval
            gross_min = int((run - ctx.interval) / pd.Timedelta(minutes=1))
            # Missing minutes are [start_et, end_et): pass end_et itself — the
            # closure range is left-inclusive, so subtracting one interval here
            # would drop the hole's final minute from the closure count.
            net_min = gross_min - expected_closed_minutes(start_et, end_et, spec)
            if net_min < net_threshold_min:
                continue
            is_holiday = (start_ts.date() in holidays) or (end_ts.date() in holidays)
            gap_windows.append({
                "start": str(start_ts),
                "end": str(end_ts - ctx.interval),
                "minutes": int(net_min),
                "holiday": bool(is_holiday),
            })
        ctx.extras["gap_windows"] = gap_windows

        suspect = [g for g in gap_windows if not g["holiday"]]
        legit = [g for g in gap_windows if g["holiday"]]
        if suspect:
            largest = max(suspect, key=lambda g: g["minutes"])
            findings.append(
                Finding(
                    id="CA-104",
                    severity=Severity.WARN,
                    summary=(
                        f"{len(suspect)} non-holiday data hole(s) ≥ {net_threshold_min} net missing minutes "
                        f"(largest: {largest['minutes']}min @ {largest['start']})"
                    ),
                    evidence={
                        "holes": len(suspect),
                        "largest_minutes": largest["minutes"],
                        "holiday_holes_excluded": len(legit),
                        "calendar": spec.name,
                    },
                    affected=[g["start"] for g in suspect[:20]],
                    remediation="verify vendor coverage; holes bias session-level statistics (see gaps_manifest())",
                )
            )
        if legit:
            findings.append(
                Finding(
                    id="CA-104",
                    severity=Severity.INFO,
                    summary=f"{len(legit)} hole(s) in holiday windows — legitimate closures",
                    evidence={"holes": len(legit)},
                )
            )

    findings.append(
        Finding(
            id="CA-104",
            severity=Severity.INFO,
            summary="coverage summary",
            evidence={
                "bars": int(len(df)),
                "calendar": spec.name,
                "weekend_gaps": int((diffs > pd.Timedelta(hours=24)).sum()),
                "gross_gap_candidates": int(len(gross_runs)),
                "net_holes": len(gap_windows),
            },
        )
    )
    return findings
