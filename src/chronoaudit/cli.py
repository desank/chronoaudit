"""chronoaudit CLI.

    chronoaudit audit data.csv --tz UTC --stamping open --calendar FX245 --json report.json

Exit codes: 0 = PASS, 1 = WARN, 2 = FAIL (worst finding severity),
3 = input/usage error (audit could not run), 4 = internal error.
A crash never exits 1/2 — CI must be able to tell "warning" from "broken".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chronoaudit.audit import audit
from chronoaudit.core.models import CHRONOAUDIT_VERSION, DeclaredConvention


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronoaudit",
        description="Timestamp forensics for trading data",
        epilog="exit codes: 0 PASS · 1 WARN · 2 FAIL · 3 input error · 4 internal error",
    )
    parser.add_argument("--version", action="version", version=f"chronoaudit {CHRONOAUDIT_VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="audit one dataset against its declared conventions")
    p_audit.add_argument("path", type=Path)
    p_audit.add_argument("--tz", default="UTC", help="declared timestamp zone (IANA)")
    p_audit.add_argument("--stamping", default="open", choices=["open", "close"])
    p_audit.add_argument("--calendar", default="FX245",
                         help="session calendar: FX245 or CME245 (others: session checks skip loudly)")
    p_audit.add_argument("--session-dst", default=None, choices=["us", "eu", "none"],
                         help="which DST regime the MARKET's sessions follow; only needed "
                              "when --calendar is not a supported built-in")
    p_audit.add_argument("--checks", default=None, help="comma-separated check ids subset")
    p_audit.add_argument("--json", type=Path, default=None, help="write JSON report here")
    p_audit.add_argument("--md", type=Path, default=None, help="write Markdown report here")
    p_audit.add_argument("--gaps-manifest", type=Path, default=None,
                         help="write engine-consumable exclusion-windows JSON here")

    p_align = sub.add_parser("align", help="CA-107: cross-feed alignment between two datasets of one instrument")
    p_align.add_argument("path_a", type=Path)
    p_align.add_argument("path_b", type=Path)
    p_align.add_argument("--tz-a", default="UTC")
    p_align.add_argument("--tz-b", default="UTC")
    p_align.add_argument("--stamping-a", default="open", choices=["open", "close"])
    p_align.add_argument("--stamping-b", default="open", choices=["open", "close"])
    p_align.add_argument("--max-lag-bars", type=int, default=180)
    p_align.add_argument("--window-days", type=int, default=90)
    p_align.add_argument("--json", type=Path, default=None)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "align":
        from chronoaudit.align.crossfeed import align as _align

        report = _align(
            args.path_a, args.path_b,
            declared_a=DeclaredConvention(tz=args.tz_a, stamping=args.stamping_a),
            declared_b=DeclaredConvention(tz=args.tz_b, stamping=args.stamping_b),
            max_lag_bars=args.max_lag_bars,
            window_days=args.window_days,
        )
        if args.json:
            args.json.write_text(report.to_json(), encoding="utf-8")
        print(report.to_markdown())
        return report.exit_code

    if args.command == "audit":
        declared = DeclaredConvention(tz=args.tz, stamping=args.stamping,
                                      session_calendar=args.calendar, session_dst=args.session_dst)
        checks = args.checks.split(",") if args.checks else None
        report = audit(args.path, declared=declared, checks=checks)
        if args.json:
            args.json.write_text(report.to_json(), encoding="utf-8")
        if args.md:
            args.md.write_text(report.to_markdown(), encoding="utf-8")
        if args.gaps_manifest:
            import json as _json

            args.gaps_manifest.write_text(
                _json.dumps(report.gaps_manifest(instrument=args.path.stem), indent=1),
                encoding="utf-8",
            )
        print(report.to_markdown())
        return report.exit_code
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles often default to cp1252; report text is UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # non-reconfigurable stream (pipes in tests)
            pass

    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except FileNotFoundError as exc:
        print(f"chronoaudit: error: file not found: {getattr(exc, 'filename', None) or exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"chronoaudit: error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 — a crash must never exit 1 (= WARN) in CI
        print(f"chronoaudit: internal error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
