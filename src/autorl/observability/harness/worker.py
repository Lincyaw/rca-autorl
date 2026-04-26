"""Worker tick: consume new turns from inbox, emit events and (optionally) a reminder.

Each tick is wrapped in a per-session ``flock`` so a cron-launched worker and a
long-running daemon can coexist without racing on the cursor.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detector import detect_drift
from .schema import Reminder, Verdict
from .store import HarnessStore, _Cursor
from .summarizer import summarize_turns

_DEFAULT_CONFIDENCE_THRESHOLD = 0.6
_DEFAULT_MIN_REMINDER_GAP = 5  # design doc §5: ≤ 1 reminder / N turns, N=5


@dataclass(frozen=True)
class TickResult:
    new_event_count: int
    last_turn_index: int
    verdict: Verdict
    reminder_written: bool
    suppressed_by_rate_limit: bool = False


def tick(
    store: HarnessStore,
    sid: str,
    *,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    min_reminder_gap: int = _DEFAULT_MIN_REMINDER_GAP,
) -> TickResult:
    """Process one batch of new turns for ``sid``. Idempotent across runs."""

    with store.session_lock(sid):
        return _tick_locked(
            store,
            sid,
            confidence_threshold=confidence_threshold,
            min_reminder_gap=min_reminder_gap,
        )


def _tick_locked(
    store: HarnessStore,
    sid: str,
    *,
    confidence_threshold: float,
    min_reminder_gap: int,
) -> TickResult:
    cursor = store.read_cursor(sid)
    all_turns = store.read_inbox(sid)
    new_turns = [t for t in all_turns if t.index > cursor.last_turn_index]
    if not new_turns:
        return TickResult(
            new_event_count=0,
            last_turn_index=cursor.last_turn_index,
            verdict=Verdict(drift=False),
            reminder_written=False,
        )

    prior_events = store.read_events(sid)
    new_events = summarize_turns(new_turns, prior_events, cursor.next_event_id)
    store.append_events(sid, new_events)

    all_events = prior_events + new_events
    verdict = detect_drift(all_events)

    new_last_index = new_turns[-1].index
    reminder_written = False
    suppressed = False
    next_last_reminder_index = cursor.last_reminder_at_index

    if (
        verdict.drift
        and verdict.type is not None
        and verdict.confidence >= confidence_threshold
    ):
        gap_since_last = new_last_index - cursor.last_reminder_at_index
        if cursor.last_reminder_at_index >= 0 and gap_since_last < min_reminder_gap:
            suppressed = True
        elif store.reminder_path(sid).exists():
            # A prior reminder is still queued; don't stack a second one.
            pass
        else:
            store.write_reminder(
                Reminder(
                    session_id=sid,
                    type=verdict.type,
                    confidence=verdict.confidence,
                    text=verdict.reminder,
                    created_at_event_id=len(all_events) - 1,
                )
            )
            reminder_written = True
            next_last_reminder_index = new_last_index

    store.write_cursor(
        sid,
        _Cursor(
            last_turn_index=new_last_index,
            next_event_id=cursor.next_event_id + len(new_events),
            last_reminder_at_index=next_last_reminder_index,
        ),
    )

    return TickResult(
        new_event_count=len(new_events),
        last_turn_index=new_last_index,
        verdict=verdict,
        reminder_written=reminder_written,
        suppressed_by_rate_limit=suppressed,
    )
