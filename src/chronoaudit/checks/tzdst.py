"""CA-101 (timezone/DST regime inference vs declaration) and CA-102 (DST-transition audit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronoaudit.calendars.fx import weekend_gaps
from chronoaudit.calendars.sessions import resolve_dst_family
from chronoaudit.checks.base import AuditContext, register
from chronoaudit.core.models import Finding, Severity
from chronoaudit.infer.dst import (
    DEFAULT_TZ_CANDIDATES,
    hourly_activity_profile,
    is_fixed_offset,
    profile_shift_hours,
    score_tz_candidates,
    us_dst_mask,
)

MIN_GAPS_FOR_FINGERPRINT = 4
MIN_REGIME_DAYS = 5
CONFIDENCE_MARGIN = 0.05  # best must beat declared by this much to call a mismatch
TIE_EPSILON = 0.02  # scores this close are one equivalence class, not a ranking


@register("CA-101")
def tz_regime_inference(df: pd.DataFrame, ctx: AuditContext) -> list[Finding]:
    findings: list[Finding] = []
    declared_tz = ctx.declared.tz
    naive = df.index.tz is None

    # Neither fingerprint always has discriminating power. Track whether
    # EITHER of them actually constrained the answer, so a sample that could
    # not be checked is never reported as a clean PASS (see the coverage
    # verdict at the end of this function).
    fp1_discriminates = False
    fp2_decided = False

    # ---- Fingerprint 1: FX weekend edges (24/5 data only) ----
    # Runs on wall-clock values even for tz-aware indexes: an aware label
    # can still be a lie (vendor localized the wrong wall times), and the
    # fingerprint is exactly the test that catches it.
    if ctx.declared.session_calendar == "FX245":
        work_index = df.index if naive else df.index.tz_convert(declared_tz).tz_localize(None)
        gaps = weekend_gaps(work_index, ctx.interval)
        if len(gaps) >= MIN_GAPS_FOR_FINGERPRINT:
            candidates = list(dict.fromkeys([declared_tz, *DEFAULT_TZ_CANDIDATES]))
            scores = score_tz_candidates(gaps, candidates)
            best_tz = max(scores, key=scores.get)  # type: ignore[arg-type]
            best_score = scores[best_tz]

            # Everything within TIE_EPSILON of the best is indistinguishable
            # by this test. This matters because `candidates` is led by the
            # DECLARED zone and max() returns the FIRST maximum — so on a tie
            # the "inference" is just the user's own claim handed back. A
            # winter-only FX sample ties all six candidates at 1.00, which
            # previously rendered as "inferred tz=<declared> (conf 1.00)".
            tied = [tz for tz, s in scores.items() if best_score - s <= TIE_EPSILON]
            declared_in_tie = declared_tz in tied
            # A tie is harmless when its members agree on DST behaviour (UTC
            # vs Etc/GMT-2 are provably indistinguishable from bar structure
            # and behave identically for session work). It is NOT harmless
            # when it mixes fixed-offset and DST-following zones: those agree
            # today and diverge by an hour at the next transition.
            mixed_dst = len({is_fixed_offset(tz) for tz in tied}) > 1

            ctx.inferred.tz_candidates = {k: round(v, 4) for k, v in scores.items()}
            ctx.inferred.tz_tied = tied
            ctx.inferred.tz_tie_mixes_dst = mixed_dst
            ctx.inferred.tz = declared_tz if declared_in_tie else best_tz
            ctx.inferred.tz_confidence = best_score
            ctx.inferred.dst_regime = (
                "undetermined" if mixed_dst
                else "none" if is_fixed_offset(ctx.inferred.tz) else "dst-following"
            )

            declared_score = scores.get(declared_tz, 0.0)
            if best_score - declared_score > CONFIDENCE_MARGIN:
                fp1_discriminates = True
                findings.append(
                    Finding(
                        id="CA-101",
                        severity=Severity.FATAL,
                        summary=(
                            f"weekend-edge fingerprint contradicts declared tz={declared_tz}: "
                            f"edges are constant under {best_tz} "
                            f"(score {best_score:.3f} vs {declared_score:.3f})"
                        ),
                        evidence={"scores": ctx.inferred.tz_candidates, "n_weekends": len(gaps)},
                        remediation=(
                            f"re-interpret timestamps as {best_tz} (or confirm vendor docs); "
                            "all session/time-of-day filters built on the declared tz are corrupt"
                        ),
                    )
                )
            elif len(tied) > 1:
                fp1_discriminates = not mixed_dst
                findings.append(
                    Finding(
                        id="CA-101",
                        severity=Severity.INFO,
                        summary=(
                            f"weekend-edge fingerprint consistent with declared tz={declared_tz}, "
                            f"but cannot single it out: {len(tied)} candidates tie at "
                            f"{best_score:.3f} ({', '.join(tied)})"
                            + ("" if mixed_dst else " — all with the same DST behaviour")
                        ),
                        evidence={
                            "scores": ctx.inferred.tz_candidates,
                            "tied": tied,
                            "tie_mixes_dst_behaviour": mixed_dst,
                            "n_weekends": len(gaps),
                        },
                    )
                )
            else:
                fp1_discriminates = True
                findings.append(
                    Finding(
                        id="CA-101",
                        severity=Severity.INFO,
                        summary=f"weekend-edge fingerprint consistent with declared tz={declared_tz}",
                        evidence={"scores": ctx.inferred.tz_candidates, "n_weekends": len(gaps)},
                    )
                )

    # ---- Fingerprint 2: intraday activity-profile shift across US-DST regimes ----
    # The verdict logic depends on the MARKET's DST family, not just the
    # declared stamping zone: a Tokyo-style market (no DST) shows shift 0 in
    # ANY stamping zone, so shift 0 proves nothing there. We only issue a
    # verdict when the family is known; otherwise we report the observation
    # and ask the user to confirm — never guess.
    family = resolve_dst_family(ctx.declared)
    dst_m = us_dst_mask(df.index)
    std_days = len(set(df.index.date[~dst_m]))
    dst_days = len(set(df.index.date[dst_m]))
    if std_days >= MIN_REGIME_DAYS and dst_days >= MIN_REGIME_DAYS:
        p_std = hourly_activity_profile(df, ~dst_m)
        p_dst = hourly_activity_profile(df, dst_m)
        shift, corr = profile_shift_hours(p_std, p_dst)
        evid = {
            "observed_shift_h": shift,
            "profile_corr": round(corr, 3),
            "std_days": std_days,
            "dst_days": dst_days,
            "session_dst_family": family or "unknown",
        }
        if corr <= 0.6:
            findings.append(
                Finding(id="CA-101", severity=Severity.INFO,
                        summary="activity profile too unstable across US-DST regimes to fingerprint",
                        evidence=evid)
            )
        elif family == "us":
            # US-session market: fixed-offset stamping puts US events 1h
            # EARLIER in raw clock during DST → shift = -1; DST-following
            # market-local stamping → shift = 0.
            expected = -1 if is_fixed_offset(declared_tz) else 0
            evid["expected_shift_h"] = expected
            # This fingerprint separates fixed-offset from DST-following
            # stamping directly, which is exactly the distinction a mixed
            # fingerprint-1 tie leaves open — so it resolves the ambiguity.
            fp2_decided = True
            if shift in (-1, 0):
                ctx.inferred.dst_regime = "none" if shift == -1 else "dst-following"
            if shift != expected:
                findings.append(
                    Finding(
                        id="CA-101",
                        severity=Severity.FATAL,
                        summary=(
                            f"intraday activity profile shifts {shift:+d}h across US-DST boundary; "
                            f"declared tz={declared_tz} implies {expected:+d}h — "
                            "timestamps are not in the declared zone"
                        ),
                        evidence=evid,
                        remediation="identify true stamping zone before any time-of-day filtering",
                    )
                )
            else:
                findings.append(
                    Finding(id="CA-101", severity=Severity.INFO,
                            summary="activity-profile DST shift consistent with declared tz",
                            evidence=evid)
                )
        elif family == "none":
            # DST-free market: the raw-clock profile should not move if the
            # timestamps are in a fixed-offset zone, and should move if they
            # are in a DST-following one. Movement is suspicious but not
            # proof of a tz lie, so this is a verdict at WARN, never FATAL.
            fp2_decided = True
            if shift != 0:
                findings.append(
                    Finding(
                        id="CA-101",
                        severity=Severity.WARN,
                        summary=(
                            f"activity profile shifts {shift:+d}h across US-DST boundaries although "
                            "the market is declared DST-free — check for mixed feeds or a session change"
                        ),
                        evidence=evid,
                    )
                )
            else:
                findings.append(
                    Finding(id="CA-101", severity=Severity.INFO,
                            summary="activity profile stable across US-DST regimes, consistent with a DST-free market",
                            evidence=evid)
                )
        else:
            # family unknown (or "eu", whose transition audit is not implemented
            # yet): evidence only, no verdict.
            findings.append(
                Finding(
                    id="CA-101",
                    severity=Severity.INFO,
                    summary=(
                        f"observed raw-clock profile shift {shift:+d}h across US-DST boundaries; "
                        "NO VERDICT — the market's DST family is unknown. If this market's "
                        "sessions follow US DST, set session_dst='us'; if it observes no DST "
                        "(Tokyo, India, crypto), set session_dst='none'."
                    ),
                    evidence=evid,
                )
            )

    # ---- Coverage verdict: did EITHER fingerprint actually constrain the tz? ----
    # Both can silently decline. Fingerprint 1 ties every candidate on a
    # short sample; fingerprint 2 needs >=MIN_REGIME_DAYS in BOTH US-DST
    # regimes, so a winter-only or summer-only sample skips it entirely.
    # When both decline, nothing has checked the declaration and the audit
    # must not return a clean PASS on it — that would confirm the user's own
    # guess back to them, which is the exact class of defect this tool exists
    # to catch. Found 2026-09-12 on a real January-only GBPUSD file: declared
    # Europe/London and declared UTC both returned PASS at "conf 1.00".
    if not fp1_discriminates and not fp2_decided:
        reasons = []
        if ctx.declared.session_calendar != "FX245":
            reasons.append("weekend-edge fingerprint needs the FX245 calendar")
        elif not ctx.inferred.tz_candidates:
            reasons.append(
                f"weekend-edge fingerprint needs >={MIN_GAPS_FOR_FINGERPRINT} weekend gaps"
            )
        else:
            reasons.append(
                f"weekend-edge fingerprint ties {len(ctx.inferred.tz_tied)} zones that "
                "disagree on DST behaviour"
            )
        reasons.append(
            f"activity-profile fingerprint needs >={MIN_REGIME_DAYS} days in BOTH US-DST "
            "regimes and a known session DST family"
        )
        findings.append(
            Finding(
                id="CA-101",
                severity=Severity.WARN,
                summary=(
                    f"timezone NOT VERIFIED — neither fingerprint could discriminate, so "
                    f"declared tz={declared_tz} is unchecked, not confirmed"
                ),
                evidence={
                    "why": reasons,
                    "tied": ctx.inferred.tz_tied,
                    "scores": ctx.inferred.tz_candidates,
                },
                remediation=(
                    "audit a sample that spans a US-DST boundary (mid-March or early "
                    "November) — an unverified zone silently corrupts every session and "
                    "time-of-day filter built on it"
                ),
            )
        )
    return findings


@register("CA-102")
def dst_transition_audit(df: pd.DataFrame, ctx: AuditContext) -> list[Finding]:
    """Bar-count anomalies on Tue–Thu near each US-DST boundary.

    Tue–Thu only: weekend-adjacent sessions legitimately change length in
    fixed-offset data (the edge moves), so Mon/Fri/Sun counts are regime-
    dependent by construction. Midweek counts must be constant; a midweek
    day near a boundary missing ~an hour of bars is the classic corruption.
    """
    findings: list[Finding] = []
    family = resolve_dst_family(ctx.declared)
    if family != "us":
        # This check audits US-DST transitions; for other session families it
        # would count the wrong boundaries. Guards in audit() announce the
        # skip for unsupported calendars; an explicit family override gets a
        # one-line note here.
        return [Finding(
            id="CA-102", severity=Severity.INFO,
            summary=f"CA-102 skipped: market session DST family is '{family or 'unknown'}' — this check audits US-DST transitions only",
        )]
    idx = df.index
    dst_m = us_dst_mask(idx)
    if dst_m.all() or not dst_m.any():  # single-regime dataset: no boundary to audit
        return findings

    # Boundary dates: where consecutive calendar days change regime.
    days = pd.Series(dst_m, index=idx).groupby(idx.date).first()
    flips = days[days.ne(days.shift()).fillna(False)]
    boundaries = list(flips.index)[1:] if len(flips) > 1 else []

    if not boundaries:
        return findings

    counts = pd.Series(1, index=idx).groupby(idx.date).sum()
    weekday = pd.Series([pd.Timestamp(d).dayofweek for d in counts.index], index=counts.index)
    midweek = counts[weekday.isin([1, 2, 3])]
    if midweek.empty:
        return findings
    modal = int(midweek.mode().iloc[0])
    tol_bars = max(1, int(pd.Timedelta(minutes=45) / ctx.interval))

    anomalies = []
    for b in boundaries:
        b_ts = pd.Timestamp(b)
        window = midweek[
            (pd.DatetimeIndex(midweek.index) >= b_ts - pd.Timedelta(days=3))
            & (pd.DatetimeIndex(midweek.index) <= b_ts + pd.Timedelta(days=3))
        ]
        for day, n in window.items():
            if abs(int(n) - modal) >= tol_bars:
                anomalies.append((str(day), int(n)))

    if anomalies:
        findings.append(
            Finding(
                id="CA-102",
                severity=Severity.WARN,
                summary=(
                    f"{len(anomalies)} midweek session(s) near DST boundaries deviate "
                    f"≥{tol_bars} bars from the modal midweek count ({modal})"
                ),
                evidence={"modal_midweek_bars": modal, "deviating_days": anomalies[:10]},
                affected=[a[0] for a in anomalies[:20]],
                remediation="inspect these sessions bar-by-bar; typical causes: dropped or duplicated hour at transition",
            )
        )
    return findings
