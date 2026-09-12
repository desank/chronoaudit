"""CA-103: bar-stamping inference (open- vs close-stamped bars).

Stamping is pure relabeling, so it is only detectable against an EXTERNAL
anchor. v1 uses two anchors, both FX-specific and heuristic (hence WARN,
never FATAL, on contradiction):

1. The canonical FX session edge (17:00 ET): the first bar after the
   weekend stamps 17:00 under open-stamping, 17:01 under close-stamping.
2. The NY-open activity burst (09:30 ET): the burst bar stamps 09:30 under
   open-stamping, 09:31 under close-stamping.

Both anchors are votes; confidence = weighted agreement. Only meaningful
when CA-101 found the data ET-edge-consistent — a vendor with fixed-UTC
edges breaks anchor 1, so we skip it and lower confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronoaudit.calendars.fx import FX_EDGE_HOUR_ET
from chronoaudit.checks.base import AuditContext, register
from chronoaudit.core.models import Finding, Severity, STAMPING_CLOSE, STAMPING_OPEN

CONFIDENCE_WARN = 0.8


def _to_et(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex | None:
    try:
        aware = index.tz_localize(tz, nonexistent="NaT", ambiguous="NaT") if index.tz is None else index
        return aware.tz_convert("America/New_York")
    except Exception:
        return None


@register("CA-103")
def stamping_inference(df: pd.DataFrame, ctx: AuditContext) -> list[Finding]:
    if ctx.declared.session_calendar != "FX245":
        return []
    tz = ctx.inferred.tz if ctx.inferred.tz_confidence >= 0.9 else ctx.declared.tz
    et = _to_et(df.index, tz)
    if et is None:
        return []
    interval_min = ctx.interval / pd.Timedelta(minutes=1)

    open_votes = 0.0
    close_votes = 0.0
    n_anchors = 0

    # Anchor 1: week-open edge minute across weekends (positional — no get_loc,
    # which breaks on duplicate timestamps).
    naive_for_gaps = df.index if df.index.tz is None else df.index.tz_localize(None)
    vals = naive_for_gaps.as_unit("ns").asi8
    pos_after = np.where(np.diff(vals) > int(pd.Timedelta(hours=24).value))[0] + 1
    if len(pos_after) >= 4:
        first_et = et[pos_after]
        minutes = first_et.hour * 60 + first_et.minute
        edge = FX_EDGE_HOUR_ET * 60
        open_hits = float(np.mean(minutes == edge))
        close_hits = float(np.mean(minutes == edge + interval_min))
        if open_hits + close_hits > 0.5:  # edges must be ET-canonical for this anchor
            open_votes += open_hits
            close_votes += close_hits
            n_anchors += 1

    # Anchor 2: NY-open activity burst minute.
    if "volume" in df.columns:
        minute_of_day = et.hour * 60 + et.minute
        prof = df["volume"].groupby(minute_of_day).median()
        ny = 9 * 60 + 30
        window = prof.reindex(range(ny - 5, ny + 6)).dropna()
        if len(window) >= 5:
            burst = int(window.idxmax())
            if burst == ny:
                open_votes += 1.0
                n_anchors += 1
            elif burst == ny + interval_min:
                close_votes += 1.0
                n_anchors += 1

    if n_anchors == 0:
        return [Finding(id="CA-103", severity=Severity.INFO,
                        summary="stamping inference inconclusive (no usable anchors)")]

    total = open_votes + close_votes
    if total == 0:
        return [Finding(id="CA-103", severity=Severity.INFO,
                        summary="stamping inference inconclusive (anchors off-canonical)")]

    inferred = STAMPING_OPEN if open_votes >= close_votes else STAMPING_CLOSE
    confidence = max(open_votes, close_votes) / total
    if n_anchors < 2:
        # One heuristic anchor is not proof: e.g. a tick-volume CFD whose max
        # burst minute is 09:31 for microstructure reasons would otherwise
        # claim close-stamping at conf 1.0 while cross-feed lag says open.
        confidence *= 0.75
    ctx.inferred.stamping = inferred
    ctx.inferred.stamping_confidence = confidence

    evid = {
        "open_votes": round(open_votes, 3),
        "close_votes": round(close_votes, 3),
        "anchors_used": n_anchors,
        "tz_used": tz,
    }
    if inferred != ctx.declared.stamping and confidence >= CONFIDENCE_WARN:
        return [Finding(
            id="CA-103",
            severity=Severity.WARN,
            summary=(
                f"bars appear {inferred}-stamped (conf {confidence:.2f}) but declared "
                f"{ctx.declared.stamping}-stamped — a mislabel here is a one-bar look-ahead "
                "for every downstream consumer"
            ),
            evidence=evid,
            remediation="verify against vendor docs and a known event minute; fix the declaration or shift the index",
        )]
    if inferred != ctx.declared.stamping:
        # Contradiction below the WARN bar: say so honestly — an INFO that
        # claims "consistent" while the inference disagrees is a false PASS.
        return [Finding(id="CA-103", severity=Severity.INFO,
                        summary=(f"stamping inferred {inferred} (conf {confidence:.2f}) CONTRADICTS "
                                 f"declared {ctx.declared.stamping}, but confidence is below the "
                                 f"{CONFIDENCE_WARN} warning bar — verify against vendor docs"),
                        evidence=evid)]
    return [Finding(id="CA-103", severity=Severity.INFO,
                    summary=f"stamping inferred {inferred} (conf {confidence:.2f}), consistent with declaration",
                    evidence=evid)]
