"""Core data models for chronoaudit.

The central design idea: every check compares what the vendor/user DECLARES
about a dataset's time conventions against what the data actually EXHIBITS
(inferred), or against a reference (exchange calendar, second feed).
Silent conventions are the enemy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from enum import IntEnum
from typing import Any

# Single source of truth for the runtime version (a test pins pyproject to it).
CHRONOAUDIT_VERSION = "0.1.0"


class Severity(IntEnum):
    """Finding severity. CLI exit code = worst severity found (0/1/2)."""

    INFO = 0    # convention notes, statistics
    WARN = 1    # biased or suspicious — review before trusting results
    FATAL = 2   # results built on this data are artifacts — do not backtest

    def __str__(self) -> str:  # pragma: no cover
        return self.name


# Stamping conventions: does the bar timestamp mark the bar's open or close?
STAMPING_OPEN = "open"
STAMPING_CLOSE = "close"
STAMPING_UNKNOWN = "unknown"


@dataclass
class DeclaredConvention:
    """What the vendor documentation / user CLAIMS about the data.

    tz: IANA zone the timestamps are expressed in (e.g. "UTC",
        "America/New_York") or a fixed offset ("Etc/GMT-2" for an
        MT5-style UTC+2 broker with no DST — note Etc zones invert sign).
    stamping: "open" or "close" — which bar edge the timestamp marks.
    session_calendar: exchange_calendars code ("XCME", "XNYS", ...) or
        the built-in "FX245" 24/5 forex calendar.
    bar_interval: expected bar spacing; None = infer from data.
    """

    tz: str = "UTC"
    stamping: str = STAMPING_OPEN
    session_calendar: str = "FX245"
    bar_interval: timedelta | None = None
    # Which DST regime the MARKET's sessions follow: "us", "eu", or "none"
    # (= the market observes no DST, e.g. Tokyo, India, crypto). Only needed
    # when session_calendar is not a supported built-in — built-ins imply it.
    # When this is unknown, DST-based checks report evidence WITHOUT a
    # verdict rather than assume a regime.
    session_dst: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bar_interval"] = str(self.bar_interval) if self.bar_interval else None
        return d


@dataclass
class InferredConvention:
    """What the data actually exhibits (filled in by inference engines)."""

    tz: str | None = None
    tz_confidence: float = 0.0
    tz_candidates: dict[str, float] = field(default_factory=dict)
    # Candidates the weekend-edge fingerprint cannot separate from the best.
    # More than one entry means the fingerprint identified an EQUIVALENCE
    # CLASS, not a zone — `tz` then means "the declaration is among those
    # consistent with the data", never "the data proves this zone". Reporting
    # must not present that as a confident inference; see `_tz_summary`.
    tz_tied: list[str] = field(default_factory=list)
    # True when the tied set mixes fixed-offset and DST-following zones.
    # A tie among zones that share DST behaviour (UTC vs Etc/GMT-2) is a
    # documented limit and harmless for session work; a tie that spans both
    # kinds means the sample is too short to tell them apart, and they will
    # diverge by an hour at the next transition.
    tz_tie_mixes_dst: bool = False
    stamping: str = STAMPING_UNKNOWN
    stamping_confidence: float = 0.0
    bar_interval: timedelta | None = None
    dst_regime: str | None = None  # "none" (fixed offset) | "us" | "eu" | ...

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bar_interval"] = str(self.bar_interval) if self.bar_interval else None
        return d


@dataclass
class Finding:
    """One defect or observation.

    id: stable check code ("CA-102"). affected: ISO date/timestamp strings
    locating the defect. evidence: check-specific measurements — every
    finding must carry the numbers that justify it (no bare assertions).
    """

    id: str
    severity: Severity
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    affected: list[str] = field(default_factory=list)
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d


@dataclass
class AuditReport:
    """Full result of an audit run."""

    dataset: dict[str, Any]
    declared: DeclaredConvention
    inferred: InferredConvention
    findings: list[Finding] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)  # check by-products (e.g. gap windows); not serialized
    chronoaudit_version: str = CHRONOAUDIT_VERSION

    @property
    def verdict(self) -> str:
        worst = max((f.severity for f in self.findings), default=Severity.INFO)
        return {Severity.INFO: "PASS", Severity.WARN: "WARN", Severity.FATAL: "FAIL"}[worst]

    @property
    def exit_code(self) -> int:
        return int(max((f.severity for f in self.findings), default=Severity.INFO))

    def to_dict(self) -> dict[str, Any]:
        return {
            "chronoaudit_version": self.chronoaudit_version,
            "dataset": self.dataset,
            "declared": self.declared.to_dict(),
            "inferred": self.inferred.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "verdict": self.verdict,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def gaps_manifest(self, instrument: str = "", include_holiday: bool = False) -> dict[str, Any]:
        """Engine-consumable exclusion-windows manifest from CA-104 gap data.

        Backtest engines read `gaps` and skip those windows; holiday closures
        are excluded by default (legitimate, not data loss).
        """
        windows = self.artifacts.get("gap_windows", [])
        if not include_holiday:
            windows = [w for w in windows if not w.get("holiday")]
        return {
            "instrument": instrument,
            "tz": self.declared.tz,
            "interval": self.dataset.get("interval"),
            "generated_by": f"chronoaudit {self.chronoaudit_version}",
            "note": "windows to exclude from session/time-of-day backtests",
            "gaps": windows,
        }

    def _tz_summary(self) -> str:
        """One-line tz rendering that never sells a tie as a confident inference.

        On a tie the candidate list is led by the DECLARED zone, so a naive
        `max()` hands the declaration straight back at the tied score — the
        summary line would read "inferred tz=<whatever you claimed> conf 1.00"
        on data that cannot distinguish six zones. Say "tied" instead.
        """
        inf = self.inferred
        if len(inf.tz_tied) > 1:
            others = [z for z in inf.tz_tied if z != inf.tz]
            note = (
                "NOT CONFIRMED — tied with"
                if inf.tz_tie_mixes_dst
                else "tied, same DST behaviour, with"
            )
            return (
                f"tz={inf.tz} ({note} {', '.join(others)}"
                f"; edge-constancy {inf.tz_confidence:.2f})"
            )
        return f"tz={inf.tz} (conf {inf.tz_confidence:.2f})"

    def to_markdown(self) -> str:
        lines = [
            f"# chronoaudit report — verdict: **{self.verdict}**",
            "",
            f"- rows: {self.dataset.get('rows')} · range: {self.dataset.get('start')} → {self.dataset.get('end')}"
            f" · interval: {self.dataset.get('interval')}",
            f"- declared: tz={self.declared.tz}, stamping={self.declared.stamping},"
            f" calendar={self.declared.session_calendar}",
            f"- inferred: {self._tz_summary()},"
            f" stamping={self.inferred.stamping} (conf {self.inferred.stamping_confidence:.2f}),"
            f" dst_regime={self.inferred.dst_regime}",
            "",
        ]
        if not self.findings:
            lines.append("No findings. Data is consistent with its declared conventions.")
        for f in sorted(self.findings, key=lambda x: -int(x.severity)):
            lines += [
                f"## [{f.severity.name}] {f.id} — {f.summary}",
                "",
                *(f"- affected: {', '.join(f.affected[:10])}"
                  + (f" (+{len(f.affected) - 10} more)" if len(f.affected) > 10 else "")
                  for _ in [0] if f.affected),
                *(f"- {k}: {v}" for k, v in f.evidence.items()),
                *([f"- remediation: {f.remediation}"] if f.remediation else []),
                "",
            ]
        return "\n".join(lines)
