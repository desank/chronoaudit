"""Audit orchestrator: load → infer → run checks → report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Importing check modules registers them.
import chronoaudit.checks.basic  # noqa: F401
import chronoaudit.checks.calendar_fx  # noqa: F401
import chronoaudit.checks.rangestats  # noqa: F401
import chronoaudit.checks.regime  # noqa: F401
import chronoaudit.checks.stamping  # noqa: F401
import chronoaudit.checks.tzdst  # noqa: F401
from chronoaudit.calendars.sessions import SPECS
from chronoaudit.checks.base import REGISTRY, AuditContext
from chronoaudit.core.models import AuditReport, DeclaredConvention, Finding, InferredConvention, Severity
from chronoaudit.io.loaders import infer_interval, load_ohlcv


def audit(
    source: pd.DataFrame | str | Path,
    declared: DeclaredConvention | None = None,
    checks: list[str] | None = None,
) -> AuditReport:
    """Run a full time-integrity audit.

    checks: optional subset of check ids (e.g. ["CA-101", "CA-104"]);
    default runs everything registered. CA-101 always runs first because
    later checks reuse its tz inference.
    """
    df = load_ohlcv(source)
    declared = declared or DeclaredConvention()
    interval = pd.Timedelta(declared.bar_interval) if declared.bar_interval else infer_interval(df)

    ctx = AuditContext(
        declared=declared,
        inferred=InferredConvention(bar_interval=interval.to_pytimedelta()),
        interval=interval,
    )

    selected = sorted(checks or REGISTRY.keys())
    findings = []
    session_checks = {"CA-101", "CA-102", "CA-103", "CA-104", "CA-111"}

    # Guard 1 — CA-110: unsupported session calendar. Say so loudly and skip
    # the checks that need one; a silent skip would hand the user a PASS that
    # verified almost nothing. CA-101 stays selected: it self-limits to
    # evidence-only reporting when the market's DST family is unknown.
    if declared.session_calendar not in SPECS:
        skipped = sorted((session_checks - {"CA-101"}) & set(selected))
        if skipped:
            findings.append(Finding(
                id="CA-110",
                severity=Severity.WARN,
                summary=(
                    f"session calendar '{declared.session_calendar}' is not supported in this "
                    f"version (supported: {', '.join(sorted(SPECS))}) — {', '.join(skipped)} "
                    "skipped; this audit is PARTIAL"
                ),
                evidence={"supported_calendars": sorted(SPECS), "skipped_checks": skipped},
                remediation=(
                    "declare a supported calendar, or run the remaining checks knowingly; "
                    "a calendar for your market may be on the roadmap — open an issue"
                ),
            ))
            selected = [c for c in selected if c not in skipped]

    # Guard 2 — CA-113 (INFO): interval too coarse for session structure.
    # Daily/4h+ bars carry no intraday session shape to audit; this is normal
    # data, not a defect.
    if interval >= pd.Timedelta(hours=4):
        skipped = sorted(session_checks & set(selected))
        if skipped:
            findings.append(Finding(
                id="CA-113",
                severity=Severity.INFO,
                summary=(
                    f"bar interval {interval} is too coarse for session-structure checks — "
                    f"{', '.join(skipped)} skipped (they need intraday bars, 1h or finer)"
                ),
                evidence={"interval": str(interval), "skipped_checks": skipped},
            ))
            selected = [c for c in selected if c not in skipped]
    else:
        # Guard 3 — CA-113 (WARN): sparse intraday coverage. Session-structure
        # checks are meaningless on a mostly-empty cache (seen in the wild: a
        # 1m futures cache whose modal weekday held 2 bars).
        counts = pd.Series(1, index=df.index).groupby(df.index.date).sum()
        wd = pd.Series([pd.Timestamp(d).dayofweek for d in counts.index], index=counts.index)
        midweek = counts[wd < 5]
        modal_bars = int(midweek.mode().iloc[0]) if not midweek.empty else 0
        min_expected = max(10, int(pd.Timedelta(hours=6) / interval))
        if modal_bars < min_expected and session_checks & set(selected):
            findings.append(Finding(
                id="CA-113",
                severity=Severity.WARN,
                summary=(
                    f"SPARSE intraday coverage (modal weekday bar count {modal_bars} < "
                    f"{min_expected} expected for {interval}) — session-structure checks "
                    f"({', '.join(sorted(session_checks & set(selected)))}) skipped as meaningless"
                ),
                evidence={"modal_weekday_bars": modal_bars, "threshold": min_expected},
                remediation=(
                    "common causes: vendor-side tick filtering or download settings discarding "
                    "data, a futures feed following a thin or wrong contract series, a genuinely "
                    "thin instrument, or an incomplete download — diagnose the cause before "
                    "trusting any session-level statistic from this dataset"
                ),
            ))
            selected = [c for c in selected if c not in session_checks]

    for check_id in selected:
        if check_id not in REGISTRY:
            raise ValueError(f"unknown check id {check_id}; known: {sorted(REGISTRY)}")
        findings.extend(REGISTRY[check_id](df, ctx))

    dataset = {
        "path": str(source) if isinstance(source, (str, Path)) else "<DataFrame>",
        "rows": int(len(df)),
        "start": str(df.index[0]) if len(df) else None,
        "end": str(df.index[-1]) if len(df) else None,
        "interval": str(interval),
        "tz_aware": df.index.tz is not None,
    }
    return AuditReport(dataset=dataset, declared=declared, inferred=ctx.inferred,
                       findings=findings, artifacts=ctx.extras)
