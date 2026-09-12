"""CA-112: cross-instrument corroboration of data holes.

A hole in one feed means little on its own. Diffed against a correlated
reference feed (same vendor or same market), every hole classifies as:

- instrument_specific_loss — reference traded normally: YOUR data is missing
- market_wide_dark         — reference dark too: market/vendor-wide event
- (holiday flags pass through from CA-104 — legitimate closures)

Doing this by hand is what separates genuine data loss from the low-liquidity
stretches around public holidays, where a thin or absent feed is the market
behaving normally; this automates it as a first-class check.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chronoaudit.audit import audit
from chronoaudit.core.models import AuditReport, DeclaredConvention, Finding, Severity
from chronoaudit.io.loaders import infer_interval, load_ohlcv

ACTIVE_RATIO = 0.5  # reference bars present / hole minutes above this = reference was trading


def corroborate(
    primary: pd.DataFrame | str | Path,
    reference: pd.DataFrame | str | Path,
    declared_primary: DeclaredConvention | None = None,
    declared_reference: DeclaredConvention | None = None,
) -> AuditReport:
    """Run CA-104 on the primary feed, then classify each hole against the reference."""
    declared_primary = declared_primary or DeclaredConvention()
    declared_reference = declared_reference or DeclaredConvention()

    report = audit(primary, declared=declared_primary, checks=["CA-101", "CA-104"])
    gaps = report.artifacts.get("gap_windows", [])
    if not gaps:
        report.findings.append(Finding(id="CA-112", severity=Severity.INFO,
                                       summary="no holes to corroborate"))
        return report

    ref = load_ohlcv(reference)
    ref_idx = ref.index
    if ref_idx.tz is None:
        ref_idx = ref_idx.tz_localize(declared_reference.tz)
    ref_idx = ref_idx.tz_convert("UTC").tz_localize(None)
    ref_sorted = ref_idx.sort_values()
    # Expected bar count scales with the REFERENCE's interval — a 15m reference
    # feed contributes 4 bars/hour, not 60; without this, coarse references
    # classify every hole as market-wide dark.
    ref_int_min = infer_interval(ref) / pd.Timedelta(minutes=1)

    prim_tz = declared_primary.tz
    n_loss = n_dark = 0
    for gap in gaps:
        start = pd.Timestamp(gap["start"])
        end = pd.Timestamp(gap["end"])
        if start.tzinfo is None:
            start_utc = start.tz_localize(prim_tz).tz_convert("UTC").tz_localize(None)
            end_utc = end.tz_localize(prim_tz).tz_convert("UTC").tz_localize(None)
        else:
            start_utc = start.tz_convert("UTC").tz_localize(None)
            end_utc = end.tz_convert("UTC").tz_localize(None)
        n_ref = int(ref_sorted.searchsorted(end_utc, side="right") - ref_sorted.searchsorted(start_utc, side="left"))
        expected_ref_bars = gap["minutes"] / ref_int_min
        active = n_ref >= max(1, expected_ref_bars * ACTIVE_RATIO)
        gap["reference_bars"] = n_ref
        gap["class"] = "instrument_specific_loss" if active else "market_wide_dark"
        if gap["holiday"]:
            continue
        if active:
            n_loss += 1
        else:
            n_dark += 1

    sev = Severity.WARN if n_loss else Severity.INFO
    report.findings.append(Finding(
        id="CA-112",
        severity=sev,
        summary=(
            f"hole corroboration vs reference: {n_loss} instrument-specific loss(es), "
            f"{n_dark} market-wide dark window(s) "
            f"(+{sum(1 for g in gaps if g['holiday'])} holiday, excluded)"
        ),
        evidence={
            "instrument_specific_losses": n_loss,
            "market_wide_dark": n_dark,
            "worst_loss": max(
                ({"start": g["start"], "minutes": g["minutes"]} for g in gaps
                 if g["class"] == "instrument_specific_loss" and not g["holiday"]),
                key=lambda x: x["minutes"], default=None,
            ),
        },
        remediation="re-download instrument-specific windows from the vendor, or exclude via gaps_manifest()",
    ))
    return report
