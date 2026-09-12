"""Timezone / DST inference engines.

Two independent fingerprints, both convention-free (they never assume a
specific session-edge hour — they demand *constancy under the correct
timezone interpretation*):

1. Weekend-gap fingerprint (24/5 FX data): in the timezone the data is
   truly expressed in, the weekend edges sit at a constant local wall time
   in ET across US-DST regimes. Interpreting the data in the wrong tz makes
   the ET edge time flip by 1h between winter and summer.

2. Session-profile shift (any data): the hourly activity profile, computed
   on raw timestamp hours, shifts by 1h across a US-DST boundary iff the
   data is expressed in a fixed-offset zone (UTC etc.), and stays put iff
   expressed in a DST-following market-local zone. Either can be fine —
   what matters is agreement with the DECLARED zone.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = "America/New_York"

DEFAULT_TZ_CANDIDATES = [
    "UTC",
    "America/New_York",
    "Europe/London",
    "Europe/Athens",   # EET with EU DST — common MT4/MT5 broker time
    "Etc/GMT-2",       # fixed UTC+2 (Etc zones invert sign)
    "Etc/GMT-3",       # fixed UTC+3
]


def is_fixed_offset(tz: str) -> bool:
    """True when the zone never observes DST (UTC, Etc/GMT±N)."""
    z = ZoneInfo(tz)
    jan = pd.Timestamp("2024-01-15", tz=z).utcoffset()
    jul = pd.Timestamp("2024-07-15", tz=z).utcoffset()
    return jan == jul


def _edge_times_et(edges: list[pd.Timestamp], candidate: str) -> np.ndarray:
    """Interpret naive edge timestamps in `candidate` tz, express in ET, return fractional hours."""
    out = []
    for ts in edges:
        try:
            loc = ts.tz_localize(candidate) if ts.tzinfo is None else ts.tz_convert(candidate)
        except Exception:
            continue  # nonexistent/ambiguous local time under this candidate — itself a hint
        et = loc.tz_convert(ET)
        out.append(et.hour + et.minute / 60.0)
    return np.asarray(out, dtype=float)


def _circular_concentration(hours: np.ndarray, tol_minutes: float = 10.0) -> float:
    """Fraction of hour-of-day values within ±tol_minutes of their circular mean.

    Deliberately NOT the mean-resultant-length R: two clusters 1h apart on a
    24h circle give R = cos(7.5°) ≈ 0.99, so R cannot separate a DST-split
    bimodal edge from a genuinely constant ragged edge. The within-tolerance
    fraction separates them sharply (~0.95 constant vs ~0.5 or less split).
    """
    if len(hours) == 0:
        return 0.0
    # Mode-seeking, not mean-centered: holiday early-closes (CME: 09:00/13:00
    # ET sessions) are legitimate outliers that drag a circular mean off the
    # true edge cluster and collapse the score for the CORRECT tz (found on
    # real ES/ZN data: 25% outliers → UTC scored 0.03). Instead: the best
    # fraction of edges inside ANY window of ±tol_minutes on the 24h circle.
    minutes = np.sort((hours * 60.0) % 1440.0)
    ext = np.concatenate([minutes, minutes + 1440.0])  # unwrap the circle
    width = 2.0 * tol_minutes
    counts = np.searchsorted(ext, minutes + width, side="right") - np.arange(len(minutes))
    return float(counts.max() / len(minutes))


def score_tz_candidates(
    weekend_edges: list[tuple[pd.Timestamp, pd.Timestamp]],
    candidates: list[str] | None = None,
) -> dict[str, float]:
    """Score each candidate tz by constancy of weekend-edge ET wall times.

    Uses both edges of every weekend gap (close edge and open edge),
    scored separately then averaged — vendors differ on which edge is
    ragged (late Friday ticks) so neither dominates.
    """
    candidates = candidates or DEFAULT_TZ_CANDIDATES
    closes = [c for c, _ in weekend_edges]
    opens = [o for _, o in weekend_edges]
    scores: dict[str, float] = {}
    for cand in candidates:
        r_close = _circular_concentration(_edge_times_et(closes, cand))
        r_open = _circular_concentration(_edge_times_et(opens, cand))
        scores[cand] = (r_close + r_open) / 2.0
    return scores


def us_dst_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True where the calendar date is in US daylight-saving time.

    Uses the DATE only (regime membership), so it is valid whatever tz the
    raw timestamps are actually in — off-by-hours errors cannot flip the
    regime except within hours of the boundary itself, which we exclude
    from profile computations anyway. Computed on unique dates then mapped
    (4M-row datasets must stay inside the 60s perf budget).
    """
    dates = index.date
    uniq = pd.DatetimeIndex(sorted(set(dates)))
    noon_et = (uniq + pd.Timedelta(hours=12)).tz_localize(ET)  # noon is never ambiguous/nonexistent
    lookup = {d: ts.dst() != pd.Timedelta(0) for d, ts in zip(uniq.date, noon_et)}
    return np.fromiter((lookup[d] for d in dates), dtype=bool, count=len(dates))


def hourly_activity_profile(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Median activity (volume, else bar range) by raw-timestamp hour, L1-normalized."""
    sub = df[mask]
    if sub.empty:
        return np.zeros(24)
    activity = sub["volume"] if "volume" in sub.columns else (sub["high"] - sub["low"])
    prof = activity.groupby(sub.index.hour).median().reindex(range(24), fill_value=0.0).to_numpy()
    total = prof.sum()
    return prof / total if total > 0 else prof


def profile_shift_hours(p_std: np.ndarray, p_dst: np.ndarray, max_shift: int = 3) -> tuple[int, float]:
    """Best circular shift (hours) aligning the DST-regime profile onto the standard-regime one.

    Returns (shift, correlation). Sign convention: shift = +1 means the DST
    profile sits 1h LATER in raw clock terms; shift = −1 means 1h EARLIER.
    For a US-session market recorded in a fixed-offset zone (UTC etc.), US
    events land 1h EARLIER in raw clock during DST, so the fixed-offset
    signature is shift = −1; market-local (DST-following) stamping gives 0.
    What a given shift MEANS also depends on the market's own DST family —
    see the caller in checks/tzdst.py.
    """
    best_shift, best_corr = 0, -np.inf
    for s in range(-max_shift, max_shift + 1):
        c = float(np.corrcoef(p_std, np.roll(p_dst, -s))[0, 1])
        if np.isfinite(c) and c > best_corr:
            best_shift, best_corr = s, c
    return best_shift, best_corr
