"""CLI entry: ``python -m autorl.observability.harness {ingest,tick,inject}``.

Subcommands are designed to be invoked from Claude Code hooks. They speak
JSON on stdin/stdout and exit with code 0 on success. ``inject`` writes the
reminder text (if any) to stdout — silent otherwise — so a UserPromptSubmit
hook can pipe it directly into the next prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .claude_code import (
    delta_against,
    parse_hook_payload,
    read_transcript_turns,
)
from .schema import Turn
from .store import HarnessStore
from .worker import tick

_DEFAULT_ROOT = ".harness"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autorl.observability.harness")
    parser.add_argument(
        "--root",
        default=_DEFAULT_ROOT,
        help=f"Harness storage root (default: {_DEFAULT_ROOT})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Append a transcript delta to the inbox.")
    p_ingest.add_argument(
        "--session",
        help="Session id. Required for --input mode; auto-detected for --from-hook.",
    )
    p_ingest.add_argument(
        "--input",
        default=None,
        help="Path to a JSON payload of {turns: [...]}, or '-' to read stdin.",
    )
    p_ingest.add_argument(
        "--from-hook",
        action="store_true",
        help=(
            "Treat stdin as a Claude Code hook payload; derive session_id and "
            "transcript_path from it, then ingest the inbox delta. Silent "
            "no-op if the payload is unrecognizable."
        ),
    )

    p_tick = sub.add_parser("tick", help="Run one summarize+detect pass for a session.")
    p_tick.add_argument("--session", required=True)
    p_tick.add_argument("--confidence", type=float, default=0.6)
    p_tick.add_argument(
        "--min-reminder-gap",
        type=int,
        default=5,
        help="Minimum number of turns between two reminders for the same session.",
    )

    p_inject = sub.add_parser(
        "inject",
        help="Print and consume any pending reminder for a session.",
    )
    p_inject.add_argument(
        "--session",
        help="Session id. Required for direct mode; auto-detected for --from-hook.",
    )
    p_inject.add_argument(
        "--from-hook",
        action="store_true",
        help="Read session_id from a Claude Code hook payload on stdin.",
    )

    return parser


def _read_payload(arg: str) -> dict:
    if arg == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(arg).read_text(encoding="utf-8"))


def _cmd_ingest(args: argparse.Namespace) -> int:
    store = HarnessStore(args.root)
    if args.from_hook:
        payload = parse_hook_payload(sys.stdin.read())
        if payload is None:
            return 0  # not a Claude Code payload — silent no-op
        sid = args.session or payload.session_id
        if not payload.has_transcript:
            print(json.dumps({"appended": 0, "session": sid, "reason": "no transcript"}))
            return 0
        all_turns = read_transcript_turns(payload.transcript_path)  # type: ignore[arg-type]
        existing = store.read_inbox(sid)
        delta = delta_against(existing, all_turns)
        store.append_inbox(sid, delta)
        print(json.dumps({"appended": len(delta), "session": sid}))
        return 0

    if not args.session:
        print("--session is required when not using --from-hook", file=sys.stderr)
        return 2
    source = args.input if args.input is not None else "-"
    payload = _read_payload(source)
    turns = [Turn.from_dict(t) for t in payload.get("turns", [])]
    store.append_inbox(args.session, turns)
    print(json.dumps({"appended": len(turns)}))
    return 0


def _cmd_tick(args: argparse.Namespace) -> int:
    store = HarnessStore(args.root)
    result = tick(
        store,
        args.session,
        confidence_threshold=args.confidence,
        min_reminder_gap=args.min_reminder_gap,
    )
    payload = {
        "new_events": result.new_event_count,
        "last_turn_index": result.last_turn_index,
        "drift": result.verdict.drift,
        "drift_type": result.verdict.type.value if result.verdict.type else None,
        "confidence": result.verdict.confidence,
        "reminder_written": result.reminder_written,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _cmd_inject(args: argparse.Namespace) -> int:
    store = HarnessStore(args.root)
    sid = args.session
    if args.from_hook:
        payload = parse_hook_payload(sys.stdin.read())
        if payload is None:
            return 0
        sid = sid or payload.session_id
    if not sid:
        # No session resolvable; silent no-op (matches hook fail-open behavior).
        return 0
    reminder = store.pop_reminder(sid)
    if reminder is None:
        return 0
    sys.stdout.write(reminder.text)
    if not reminder.text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "ingest":
        return _cmd_ingest(args)
    if args.cmd == "tick":
        return _cmd_tick(args)
    if args.cmd == "inject":
        return _cmd_inject(args)
    raise AssertionError(f"unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
