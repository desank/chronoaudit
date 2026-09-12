"""Built-in FX 24/5 calendar.

The FX week runs Sunday 17:00 ET to Friday 17:00 ET, continuously.
Everything between Friday 17:00 ET and Sunday 17:00 ET is the weekend gap.
The exact edge minute varies by vendor (some close 16:59, some 22:00 UTC
fixed) — checks therefore test *constancy in the right timezone*, not a
hard-coded edge hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"
FX_EDGE_HOUR_ET = 17  # canonical session edge in ET


def in_weekend_gap(ts_et: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: timestamps (ET-aware) falling inside the FX weekend gap."""
    dow = ts_et.dayofweek  # Mon=0 .. Sun=6
    hour = ts_et.hour
    fri_after_close = (dow == 4) & (hour >= FX_EDGE_HOUR_ET)
    saturday = dow == 5
    sun_before_open = (dow == 6) & (hour < FX_EDGE_HOUR_ET)
    return pd.Series(fri_after_close | saturday | sun_before_open, index=ts_et)


def fx_holiday_dates(years: range) -> set:
    """Dates on which FX markets are closed or structurally thin: Christmas /
    New Year windows and Good Friday. Used to downgrade CA-104 hole findings
    from WARN to INFO — on an 11-year real corpus, ~half of flagged holes
    were these (see ROADMAP: first production forensics)."""
    from dateutil.easter import easter

    days = set()
    for y in years:
        gf = pd.Timestamp(easter(y)) - pd.Timedelta(days=2)
        days.add(gf.date())
        for m, d in ((12, 24), (12, 25), (12, 26), (12, 31), (1, 1), (1, 2)):
            days.add(pd.Timestamp(year=y, month=m, day=d).date())
    return days


def weekend_gaps(index: pd.DatetimeIndex, interval: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(last_bar_close, first_bar_after) pairs for gaps that look like weekends (> 24h).

    Positional (numpy) implementation — get_loc() returns a slice when the
    index has duplicate timestamps, and duplicated data is exactly the kind
    of dataset this tool gets pointed at.
    """
    vals = index.as_unit("ns").asi8
    pos_after = np.where(np.diff(vals) > int(pd.Timedelta(hours=24).value))[0] + 1
    return [(index[p - 1] + interval, index[p]) for p in pos_after]
