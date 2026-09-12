"""Ingestion: normalize arbitrary OHLCV inputs to the canonical frame.

Canonical form: pandas DataFrame, sorted DatetimeIndex (tz preserved
exactly as supplied — auditing the supplied convention is the point),
lowercase columns: open, high, low, close, volume (volume optional).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_COLUMN_ALIASES = {
    "open": {"open", "o", "op"},
    "high": {"high", "h", "hi"},
    "low": {"low", "l", "lo"},
    "close": {"close", "c", "cl", "last"},
    "volume": {"volume", "vol", "v", "tickvol", "tick_volume"},
}

_TIME_ALIASES = {
    "ts", "time", "timestamp", "datetime", "date", "dt", "gmt time",
    "open_time", "open time", "close_time",  # Binance-style kline exports
}

# Epoch magnitudes for "now"-era data (1990–2100), used to infer the unit of
# integer time columns. MT5's copy_rates gives epoch seconds; Binance gives ms.
_EPOCH_UNITS = [  # (unit, lower bound, upper bound)
    ("s", 6e8, 5e9),
    ("ms", 6e11, 5e12),
    ("us", 6e14, 5e15),
    ("ns", 6e17, 5e18),
]


def _parse_time_column(col: pd.Series) -> pd.DatetimeIndex:
    """Parse a time column: epoch integers (unit inferred from magnitude)
    or datetime strings. Loud failure over silent misparse."""
    if pd.api.types.is_numeric_dtype(col):
        med = float(pd.Series(col).dropna().astype("float64").median())
        for unit, lo, hi in _EPOCH_UNITS:
            if lo <= abs(med) < hi:
                return pd.DatetimeIndex(pd.to_datetime(col, unit=unit))
        raise ValueError(
            f"time column is numeric but does not look like an epoch in any unit "
            f"(median value {med:.4g}); if these are row numbers, supply a real time column"
        )
    try:
        return pd.DatetimeIndex(pd.to_datetime(col))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"could not parse the time column: {exc}. "
            "If dates are day-first (e.g. 31.12.2024), convert them to ISO or a "
            "DatetimeIndex yourself before auditing — chronoaudit refuses to guess "
            "day/month order (a silent swap would corrupt the audit itself)."
        ) from exc


def load_ohlcv(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load and normalize an OHLCV dataset from a DataFrame, CSV, or parquet.

    Raises ValueError when no time axis or no price columns can be located —
    a dataset we cannot orient in time cannot be audited.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".parquet":
            try:
                df = pd.read_parquet(path)
            except ImportError as exc:
                raise ValueError(
                    "reading parquet needs a parquet engine: pip install 'chronoaudit[parquet]'"
                ) from exc
        elif path.suffix.lower() in {".csv", ".txt"}:
            # sep=None + python engine sniffs the delimiter — MT5 exports are
            # tab-separated, most vendors use commas, some use semicolons.
            df = pd.read_csv(path, sep=None, engine="python")
        else:
            raise ValueError(f"unsupported file type: {path.suffix}")
    else:
        df = source.copy()

    # Strip MT5-terminal-style angle brackets (<DATE>, <OPEN>, <TICKVOL> …).
    df.columns = [str(c).strip().strip("<>") for c in df.columns]

    if not isinstance(df.index, pd.DatetimeIndex):
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        # MT5 terminal exports split the axis into DATE + TIME columns
        # (DATE is dot-separated: 2024.01.15).
        if "date" in cols_lower and "time" in cols_lower:
            combined = df[cols_lower["date"]].astype(str) + " " + df[cols_lower["time"]].astype(str)
            parsed = None
            for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    parsed = pd.DatetimeIndex(pd.to_datetime(combined, format=fmt))
                    break
                except (ValueError, TypeError):
                    continue
            df.index = parsed if parsed is not None else _parse_time_column(combined)
            df = df.drop(columns=[cols_lower["date"], cols_lower["time"]])
        else:
            time_col = next((cols_lower[a] for a in cols_lower if a in _TIME_ALIASES), None)
            if time_col is None:
                raise ValueError(
                    "no DatetimeIndex and no recognizable time column "
                    f"(looked for {sorted(_TIME_ALIASES)})"
                )
            df.index = _parse_time_column(df[time_col])
            df = df.drop(columns=[time_col])

    rename: dict[str, str] = {}
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for canon, aliases in _COLUMN_ALIASES.items():
        # Canonical name first, then sorted aliases: deterministic winner when
        # a file carries both (e.g. volume AND tickvol) — sets iterate in hash order.
        for alias in (canon, *sorted(aliases - {canon})):
            if alias in lower_map:
                rename[lower_map[alias]] = canon
                break
    df = df.rename(columns=rename)

    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"missing OHLC columns after normalization: {sorted(missing)}")

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep]

    # Prices must be numeric BEFORE any check touches them — string prices
    # (e.g. European comma decimals) would otherwise fail deep inside a check.
    for col in ("open", "high", "low", "close"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.isna().sum() > df[col].isna().sum():
                bad = df[col][coerced.isna() & df[col].notna()]
                raise ValueError(
                    f"price column '{col}' is not numeric (first bad value: {bad.iloc[0]!r}). "
                    "Common cause: comma decimal separator — convert to dot-decimal before auditing."
                )
            df[col] = coerced
    if "volume" in df.columns and not pd.api.types.is_numeric_dtype(df["volume"]):
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df.index.name = "ts"
    return df.sort_index()


def infer_interval(df: pd.DataFrame) -> pd.Timedelta:
    """Modal bar spacing — robust to gaps because gaps are the minority."""
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        raise ValueError("cannot infer interval from fewer than 2 bars")
    return diffs.mode().iloc[0]
