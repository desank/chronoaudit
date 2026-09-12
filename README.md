# chronoaudit

**Timestamp forensics for trading data.** It finds the time-integrity defects that silently corrupt backtests — DST bugs, mislabeled vendor timezones, wrong bar-stamping conventions, session anomalies, cross-feed misalignment — *before* the data enters your engine.

```bash
pip install chronoaudit
chronoaudit audit gbpusd_1m.csv --tz UTC --stamping open --calendar FX245
# parquet files too: pip install 'chronoaudit[parquet]'
```

It **diagnoses; it never edits your data.** An auditor that rewrites its own evidence cannot be re-audited — and the durable fix for time defects usually lives upstream (your downloader, your vendor, your resampler), not in the file.

## Why this exists

Every seasoned quant has a version of these scars (real, anonymized incidents):

- A time-of-day filter hardcoded in UTC hours. The US session shifts by an hour twice a year; **~8 months of every backtested year were filtered wrong.**
- A breakout system whose +487R came entirely from a resample that leaked hours of future data.
- Close-stamped bars treated as open-stamped: **one full bar of look-ahead** for every consumer, invisible in any statistics.
- Two feeds of one instrument, one DST-shifting and one not: every cross-feed comparison silently off by an hour for half the year.

These artifacts pass statistical validation, because validators validate whatever the data says. The only defense is auditing the data's *time truth* first. That is the whole job of this tool.

## 1. What it does

You **declare** what the vendor claims (`tz=UTC, stamping=open, calendar=FX245`). chronoaudit measures what the data actually **exhibits** and reports every contradiction as a finding with severity (`INFO`/`WARN`/`FATAL`), evidence numbers, affected ranges, and a remediation hint.

| ID | Check | Catches |
|----|-------|---------|
| CA-101 | Timezone/DST inference | Timestamps not in the declared zone — two independent fingerprints (weekend-edge constancy; activity-profile shift across DST regimes) |
| CA-102 | DST-transition audit | Dropped/duplicated hours around US-DST transitions |
| CA-103 | Bar-stamping inference | Close-stamped bars declared open-stamped (one-bar look-ahead) |
| CA-104 | Session/calendar conformance | Phantom weekend bars; data holes (netted against real closures; holidays classified) |
| CA-105 | Index integrity | Duplicate timestamps, off-grid bars |
| CA-106 | OHLC coherence | Impossible bars, frozen feeds |
| CA-107 | Cross-feed alignment (`chronoaudit align`) | Constant lags between two feeds; the killer: **lag that flips by 60 min between DST regimes** (one feed is lying about its zone) |
| CA-108 | Resample causality (`verify_resample`) | Coarse bars built under the opposite stamping convention |
| CA-109 | Finer-bar range stats | Reference table for "is my stop smaller than intrabar noise?" |
| CA-110 | Partial-audit notice | Your declaration used a calendar this version cannot check — said loudly, never skipped silently |
| CA-111 | Session-regime change | Backtests spanning two structurally different market sessions |
| CA-112 | Hole corroboration (`corroborate`, API) | Each hole classified against a reference feed: your loss vs market-wide dark |
| CA-113 | Coverage guard | Sparse/partial caches and too-coarse intervals — flagged instead of analyzed as if they were markets |

Extras: `--gaps-manifest` writes machine-readable exclusion windows your backtest engine can consume; CLI exit codes are CI-friendly: **0 PASS · 1 WARN · 2 FAIL · 3 input error · 4 internal error** (a crash never masquerades as a warning).

```python
from chronoaudit import audit, align, verify_resample, DeclaredConvention

report = audit("gbpusd_1m.csv",
               declared=DeclaredConvention(tz="UTC", stamping="open", session_calendar="FX245"))
print(report.verdict)       # PASS | WARN | FAIL
print(report.inferred.tz)   # what the data actually looks like
report.to_json()            # stable machine-readable schema
```

## 2. Prerequisites and assumptions

- Python ≥ 3.11; pandas ≥ 2.1, numpy, python-dateutil. Parquet input needs the `[parquet]` extra.
- Input: OHLCV **bars** with a time axis — a DataFrame with a DatetimeIndex, or CSV/parquet. Recognized time columns include `time`, `timestamp`, `date`, `Gmt time`, `open_time`; epoch integers (seconds/ms/µs/ns) are detected by magnitude; MT5 terminal exports (`<DATE>`/`<TIME>`, tab-separated) load directly.
- You must **declare** the vendor's claimed convention. That is the design: a lie can only be detected against a claim. If you don't know the claim, declare your best guess and read what the fingerprints say.
- Session-structure checks need **intraday** data (1h bars or finer); the deeper fingerprints want weeks-to-months of history spanning at least one DST boundary.

## 3. Constraints it operates under

- **Built-in session calendars: FX245 and CME245** (both anchor to New York time). Any other calendar makes the audit PARTIAL — and it tells you so (CA-110).
- **Markets that don't follow US DST** (Tokyo, India, crypto): set `session_dst="none"` and the DST checks reason correctly for them. If the market's DST family is unknown, chronoaudit **reports what it observed and refuses to issue a verdict** — it asks you to confirm rather than assume.
- **A sample that cannot verify your timezone is reported as unverified, never as a pass.** Both tz fingerprints can decline: the weekend-edge test ties every candidate on a short sample, and the activity-profile test needs at least five days in *both* US-DST regimes. When neither can discriminate, CA-101 raises a WARN saying the declared zone is *unchecked, not confirmed* — audit a window spanning mid-March or early November to resolve it. A winter-only file is the common case: `UTC` and `Europe/London` are identical in January and an hour apart from late March.
- Performance: a full audit of ~4.2M rows (11 years of 1-minute FX) takes **~60–75 s** on an ordinary desktop.
- Day-first date strings (`31.12.2024`) are refused with an explanation, not guessed — a silent day/month swap would corrupt the audit itself.

## 4. What it does NOT do

- **It never fixes data.** Remediation is a hint in each finding, plus the gaps manifest.
- No tick data — bars only in this version.
- Fixed-offset zones that differ only by offset (UTC vs UTC+2) are **provably indistinguishable** from bar structure alone; ties are reported as ties, and the report names every tied zone rather than picking one. Same for two DST-following zones one hour apart (New York vs Chicago) — CA-104/CA-107 are the checks that catch those, not CA-101.
- Stamping inference (CA-103) is heuristic, FX-only in this version, and deliberately capped at WARN.
- EU-DST transition auditing is not implemented yet (observations are still reported).
- No live/streaming monitoring; no exchange-specific calendars beyond the built-ins (the `exchange_calendars` hook exists but ships unbundled).

## 5. Roadmap

- **v0.2** — crypto 24/7 calendar; day-first date handling; EU-DST session family; `corroborate` CLI subcommand; performance pass on hole-netting.
- **v0.3** — `exchange_calendars`-based per-exchange conformance; era-aware session specs (exchanges change hours).
- **Later** — tick-level checks; an opt-in `fix --emit` that writes a **new** corrected file with a provenance sidecar (never in place); venue-clock studies.

Adding a calendar for your market is a small, well-contained contribution — issues and PRs welcome.

## Design notes

- **The detector is tested, not just run**: the suite injects each corruption class into clean synthetic data and requires the right check to fire with the right severity — and clean data to stay silent. A false-positive audit tool would be its own artifact factory.
- Deterministic: same input, same report. Pure pandas/numpy.
- Engine-independent by design: no coupling to any backtest framework.

## License

Apache-2.0. Contributions welcome under DCO sign-off.
