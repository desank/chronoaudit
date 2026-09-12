"""Session calendars: structural closures per market type.

FX245:  Sun 17:00 ET → Fri 17:00 ET, no daily break.
CME245: Sun 18:00 ET → Fri 17:00 ET, daily maintenance break 17:00–18:00 ET
        (Mon–Thu). Fits CME Globex equity/rates/energy/metals — and vendors
        that mirror them (index CFDs, spot gold).

Holes are judged NET of these closures: a 61-minute gap that spans the
maintenance break is one missing minute, not sixty-one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ET = "America/New_York"


@dataclass(frozen=True)
class SessionSpec:
    name: str
    week_close: tuple[int, int]      # (dayofweek, ET hour) — market closes
    week_open: tuple[int, int]       # (dayofweek, ET hour) — market opens
    daily_break_hour: int | None     # ET hour fully closed Mon–Thu, or None
    us_market_holidays: bool         # extend holiday set beyond FX basics
    dst_family: str = "us"           # which DST regime the SESSIONS follow: "us" | "eu" | "none"


SPECS: dict[str, SessionSpec] = {
    # Both built-ins anchor their sessions to New York time, so their session
    # clock follows US DST — this is a property of the MARKET, true for a user
    # anywhere in the world.
    "FX245": SessionSpec("FX245", (4, 17), (6, 17), None, False, dst_family="us"),
    "CME245": SessionSpec("CME245", (4, 17), (6, 18), 17, True, dst_family="us"),
}


def resolve_dst_family(declared) -> str | None:
    """Which DST regime the market's sessions follow ("us", "eu", "none"),
    or None when unknown.

    Resolution order: the user's explicit `session_dst` declaration wins;
    else a supported calendar implies it; else unknown. Checks that depend
    on this MUST NOT guess when it is None — they report evidence and ask
    the user to confirm instead.
    """
    explicit = getattr(declared, "session_dst", None)
    if explicit:
        return explicit
    spec = SPECS.get(getattr(declared, "session_calendar", ""))
    return spec.dst_family if spec else None


def holiday_dates(spec: SessionSpec, years: range) -> set:
    """Dates with legitimate full/partial closures. FX: Christmas/New Year/
    Good Friday. CME245 adds the US market holidays (early-close or closed:
    MLK, Presidents, Memorial, Juneteenth, July 4, Labor, Thanksgiving+Friday)."""
    from dateutil.easter import easter

    days = set()
    for y in years:
        days.add((pd.Timestamp(easter(y)) - pd.Timedelta(days=2)).date())  # Good Friday
        for m, d in ((12, 24), (12, 25), (12, 26), (12, 31), (1, 1), (1, 2)):
            days.add(pd.Timestamp(year=y, month=m, day=d).date())
        if spec.us_market_holidays:
            def nth_weekday(month, weekday, n):
                first = pd.Timestamp(year=y, month=month, day=1)
                offset = (weekday - first.dayofweek) % 7
                return (first + pd.Timedelta(days=offset + 7 * (n - 1))).date()

            def last_weekday(month, weekday):
                last = pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(0)
                return (last - pd.Timedelta(days=(last.dayofweek - weekday) % 7)).date()

            days.add(nth_weekday(1, 0, 3))    # MLK
            days.add(nth_weekday(2, 0, 3))    # Presidents
            days.add(last_weekday(5, 0))      # Memorial
            days.add(pd.Timestamp(year=y, month=6, day=19).date())  # Juneteenth
            days.add(pd.Timestamp(year=y, month=7, day=4).date())
            days.add(nth_weekday(9, 0, 1))    # Labor
            thanksgiving = nth_weekday(11, 3, 4)
            days.add(thanksgiving)
            days.add((pd.Timestamp(thanksgiving) + pd.Timedelta(days=1)).date())
    return days


def in_structural_closure(ts_et: pd.DatetimeIndex, spec: SessionSpec) -> np.ndarray:
    """True for timestamps inside the weekend gap or the daily break."""
    dow, hour = ts_et.dayofweek, ts_et.hour
    c_dow, c_hour = spec.week_close
    o_dow, o_hour = spec.week_open
    weekend = ((dow == c_dow) & (hour >= c_hour)) | (dow == 5) | ((dow == o_dow) & (hour < o_hour))
    if spec.daily_break_hour is not None:
        weekend = weekend | ((hour == spec.daily_break_hour) & (dow <= 3))
    return np.asarray(weekend)


def expected_closed_minutes(start_et: pd.Timestamp, end_et: pd.Timestamp, spec: SessionSpec) -> int:
    """Structurally-closed minutes inside [start_et, end_et) — for netting holes."""
    if end_et <= start_et:
        return 0
    minutes = pd.date_range(start_et, end_et, freq="1min", inclusive="left")
    return int(in_structural_closure(minutes, spec).sum())
