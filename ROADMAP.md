# Roadmap

## Done since the first production sweeps (2026-07)

- ~~Holiday-calendar whitelist for CA-104~~ — shipped (FX + US market-holiday sets; holiday holes report INFO).
- ~~Session-regime-change detector~~ — shipped as CA-111 (validated against a real in-the-wild case).
- ~~Cross-instrument corroboration~~ — shipped as CA-112 / `corroborate()`.
- ~~Known-gaps manifest output~~ — shipped (`gaps_manifest()` / `--gaps-manifest`).
- CME245 session calendar (futures/index-CFD/metals; net-of-closure hole accounting).
- Sparse-cache guard (partial downloads flagged, not misanalyzed).

## Next

- **Sub-hour break windows** — SessionSpec break granularity below 1h (some index CFDs break 16:15–17:30 ET); today the residue is classified as `recurring_daily_structural` rather than netted exactly.
- **Break-window inference** — learn a vendor's actual daily closure window from the data instead of declaring it.
- **exchange_calendars conformance** — per-exchange holiday/half-day precision beyond the built-in specs.
- **Tick-level support** — v1 is bars-only by design.
- **CA-102 holiday awareness** — midweek bar-count anomalies near DST boundaries should whitelist holiday-shortened sessions the way CA-104 does.
