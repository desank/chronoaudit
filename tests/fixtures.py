"""Synthetic FX fixture generator + mutation harness.

Clean data is generated on an ET wall clock (sessions Sun 17:00 → Fri 17:00
ET, volume peaks at London and NY opens), then expressed in the requested
timestamp zone. Mutations inject the specific defects each check must catch.
The acceptance bar: every mutation detected, clean data silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"


def synth_fx(start_sunday: str = "2024-02-04", weeks: int = 8, tz: str = "UTC", seed: int = 7) -> pd.DataFrame:
    """Clean 1-minute FX-like OHLCV, naive timestamps expressed in `tz`.

    start_sunday must be a Sunday. Default range spans the 2024-03-10 US
    spring-forward boundary so both DST regimes are represented.
    """
    rng = np.random.default_rng(seed)

    et_parts = []
    for w in range(weeks):
        sun = pd.Timestamp(start_sunday) + pd.Timedelta(weeks=w)
        wall = pd.date_range(
            sun + pd.Timedelta(hours=17),
            sun + pd.Timedelta(days=5, hours=17),
            freq="1min",
            inclusive="left",
        )
        # In-session wall times never cross the 02:00 Sunday DST switch → unambiguous.
        et_parts.append(wall.tz_localize(ET))
    et_idx = et_parts[0]
    for part in et_parts[1:]:
        et_idx = et_idx.append(part)

    minute_of_day = et_idx.hour * 60 + et_idx.minute

    def peak(center_min: int, sd: float, height: float) -> np.ndarray:
        return height * np.exp(-0.5 * ((minute_of_day - center_min) / sd) ** 2)

    volume = (
        50
        + peak(3 * 60, 45, 400)       # London open 03:00 ET
        + peak(9 * 60 + 30, 30, 600)  # NY open 09:30 ET
        + rng.integers(0, 20, len(et_idx))
    ).astype(float)

    rets = rng.normal(0, 2e-4, len(et_idx))
    close = 1.25 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 1e-4, len(et_idx)))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread

    out_idx = et_idx.tz_convert(tz).tz_localize(None)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=out_idx,
    )
    df.index.name = "ts"
    return df.sort_index()


# ---------------------------------------------------------------- mutations

def mutate_regime_shift(df: pd.DataFrame, hours: int = 1) -> pd.DataFrame:
    """Shift all DST-regime rows by +hours — the 'broker that never DST-shifts
    while claiming UTC' corruption. Produces constant raw wall times."""
    from chronoaudit.infer.dst import us_dst_mask

    mask = us_dst_mask(df.index)
    new_index = df.index.to_series()
    new_index[mask] = new_index[mask] + pd.Timedelta(hours=hours)
    out = df.copy()
    out.index = pd.DatetimeIndex(new_index.values, name="ts")
    return out.sort_index()


def mutate_drop_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Remove all bars in [start, end) — a data hole."""
    keep = ~((df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end)))
    return df[keep]


def mutate_weekend_bars(df: pd.DataFrame, saturday_noon: str, n: int = 90) -> pd.DataFrame:
    """Inject n phantom bars starting Saturday noon (in the data's own tz)."""
    ts = pd.date_range(pd.Timestamp(saturday_noon), periods=n, freq="1min")
    ref = df.iloc[0]
    phantom = pd.DataFrame(
        {c: ref[c] for c in ("open", "high", "low", "close")} | {"volume": 10.0},
        index=ts,
    )
    phantom.index.name = "ts"
    return pd.concat([df, phantom]).sort_index()


def mutate_stale(df: pd.DataFrame, start: str, n: int = 45) -> pd.DataFrame:
    """Freeze the feed for n bars from `start` (zero-range identical bars)."""
    out = df.copy()
    pos = out.index.searchsorted(pd.Timestamp(start))
    px = float(out.iloc[pos]["close"])
    out.iloc[pos : pos + n, out.columns.get_indexer(["open", "high", "low", "close"])] = px
    return out


def mutate_ohlc_break(df: pd.DataFrame, at: str, n: int = 3) -> pd.DataFrame:
    """Corrupt n bars so low > high."""
    out = df.copy()
    pos = out.index.searchsorted(pd.Timestamp(at))
    lows = out["low"].to_numpy().copy()
    highs = out["high"].to_numpy()
    lows[pos : pos + n] = highs[pos : pos + n] + 0.01
    out["low"] = lows
    return out


def mutate_duplicates(df: pd.DataFrame, at: str, n: int = 5) -> pd.DataFrame:
    """Duplicate n rows starting at `at`."""
    pos = df.index.searchsorted(pd.Timestamp(at))
    return pd.concat([df, df.iloc[pos : pos + n]]).sort_index()


def synth_cme(start_sunday: str = "2024-02-04", weeks: int = 8, tz: str = "UTC", seed: int = 7) -> pd.DataFrame:
    """Clean 1-minute CME-like OHLCV: Sun 18:00 → Fri 17:00 ET with the
    daily 17:00–18:00 ET maintenance break (Mon–Thu)."""
    df = synth_fx(start_sunday=start_sunday, weeks=weeks, tz="UTC", seed=seed)
    et = df.index.tz_localize("UTC").tz_convert(ET)
    keep = ~(
        ((et.dayofweek == 6) & (et.hour == 17))                    # Sunday 17:00–18:00 pre-open
        | ((et.hour == 17) & (et.dayofweek <= 3))                  # daily break Mon–Thu
    )
    out = df[keep]
    if tz != "UTC":
        idx = out.index.tz_localize("UTC").tz_convert(tz).tz_localize(None)
        out = out.copy()
        out.index = idx
    return out


def mutate_short_session(df: pd.DataFrame, until: str, drop_hours: tuple = (17, 18, 19, 20, 21, 22, 23, 0)) -> pd.DataFrame:
    """Before `until`, remove bars in drop_hours (raw clock) — simulates a
    vendor that traded a shorter session early in the instrument's history."""
    cutoff = pd.Timestamp(until)
    early = (df.index < cutoff) & df.index.hour.isin(drop_hours)
    return df[~early]
