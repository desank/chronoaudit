"""CA-105 (index integrity) and CA-106 (OHLC coherence)."""

from __future__ import annotations

import pandas as pd

from chronoaudit.checks.base import AuditContext, register
from chronoaudit.core.models import Finding, Severity


@register("CA-105")
def index_integrity(df: pd.DataFrame, ctx: AuditContext) -> list[Finding]:
    findings: list[Finding] = []

    dup = df.index.duplicated()
    if dup.any():
        ts = df.index[dup]
        findings.append(
            Finding(
                id="CA-105",
                severity=Severity.WARN,
                summary=f"{dup.sum()} duplicate timestamps",
                evidence={"count": int(dup.sum())},
                affected=[str(t) for t in ts[:20]],
                remediation="deduplicate (keep first/last consciously) before use",
            )
        )

    # Off-grid bars: timestamps not aligned to the modal interval grid.
    # as_unit("ns") first — pandas 3.x indexes may carry us/ms resolution,
    # and a raw int64 view in the wrong unit makes every bar look off-grid.
    ns = df.index.as_unit("ns").view("int64")
    step = int(ctx.interval / pd.Timedelta(nanoseconds=1))
    if step > 0:
        rem = pd.Series(ns % step, index=df.index)
        anchor = rem.mode().iloc[0]
        off = rem != anchor
        if off.any():
            frac = off.mean()
            findings.append(
                Finding(
                    id="CA-105",
                    severity=Severity.WARN if frac < 0.01 else Severity.FATAL,
                    summary=f"{int(off.sum())} bars off the {ctx.interval} grid ({frac:.3%})",
                    evidence={"off_grid_count": int(off.sum()), "fraction": float(frac)},
                    affected=[str(t) for t in df.index[off][:20]],
                    remediation="verify vendor bar-building; mixed-grid data breaks resampling",
                )
            )
    return findings


@register("CA-106")
def ohlc_coherence(df: pd.DataFrame, ctx: AuditContext) -> list[Finding]:
    findings: list[Finding] = []

    bad = (
        (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df["high"])
    )
    if bad.any():
        frac = bad.mean()
        findings.append(
            Finding(
                id="CA-106",
                severity=Severity.FATAL if frac > 0.001 else Severity.WARN,
                summary=f"{int(bad.sum())} bars violate OHLC bounds ({frac:.4%})",
                evidence={"count": int(bad.sum()), "fraction": float(frac)},
                affected=[str(t) for t in df.index[bad][:20]],
            )
        )

    if "volume" in df.columns and (df["volume"] < 0).any():
        neg = df["volume"] < 0
        findings.append(
            Finding(
                id="CA-106",
                severity=Severity.WARN,
                summary=f"{int(neg.sum())} bars with negative volume",
                affected=[str(t) for t in df.index[neg][:20]],
            )
        )

    # Stale plateaus: long runs of zero-range identical bars = frozen feed.
    zero_range = (df["high"] == df["low"]) & (df["close"] == df["close"].shift())
    if zero_range.any():
        run_id = (~zero_range).cumsum()
        run_lengths = zero_range.groupby(run_id).sum()
        long_runs = run_lengths[run_lengths >= 30]
        if not long_runs.empty:
            starts = []
            for rid in long_runs.index[:20]:
                seg = df.index[(run_id == rid) & zero_range]
                if len(seg):
                    starts.append(str(seg[0]))
            findings.append(
                Finding(
                    id="CA-106",
                    severity=Severity.WARN,
                    summary=f"{len(long_runs)} stale-price plateaus (≥30 identical zero-range bars) — possible feed freeze",
                    evidence={"plateaus": int(len(long_runs)), "longest": int(long_runs.max())},
                    affected=starts,
                    remediation="cross-check against a second feed for these windows",
                )
            )
    return findings
