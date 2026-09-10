"""Read-only HTTP over the session logs on disk, for the trajectory dashboard.

    uvicorn autorl.web:app --port 8000
    # from a laptop: ssh -L 8000:127.0.0.1:8000 <host>

Serves what `autorl.trajectory` projects, and nothing else: no run is started
from here, no file is written, and the scoring is the trainer's own
(`difficulty.weighted_score` through `trajectory.load_episode`) rather than a
second implementation. The dashboard is four views over these five endpoints.

The list endpoint is the one with a cost constraint. A session log is up to a
couple of megabytes and a K=8 sweep over the held-out split leaves 360 of them,
so `/episodes` reads and scores every one on request. It is cached by the home's
newest mtime: repeated page loads are free, and a finished run invalidates once.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from autorl.trajectory import (
    SESSION_FILE,
    Episode,
    case_dir_for,
    case_of,
    find_sessions,
    group_stats,
    load_episode,
    session_paths,
)

#: Where to look for Harness homes. A home is any directory with `sessions/`.
RUNS_ROOT = Path(os.getenv("RCA_RUNS_ROOT", ".runs")).expanduser()
#: Where the corpus cases live, for the ground truth a score needs. Its own
#: variable, not `RCA_DATASET_ROOT`: the training path sets that one to the
#: corpus root (`datapacks/ops-lite`, per README) and `agent.resolve_data_dir`
#: tries both it and its `cases/` child. Reading it here as the cases root would
#: resolve `datapacks/ops-lite/batch-01KQ...`, find nothing, and score every
#: episode 0.0 with no axes -- which renders exactly like a real negative result.
CASES_ROOT = Path(os.getenv("RCA_CASES_ROOT", "datapacks/ops-lite/cases")).expanduser()
#: 0.6 x 32768, the `thresholdRatio` in `agent/rca-harness/cordis.patch.yml`. A
#: peak above it is a request that went out carrying more than the meter
#: intended, which is how an episode reaches the window despite compacting.
COMPACTION_THRESHOLD = 19661

app = FastAPI(title="RCA trajectory dashboard", docs_url="/api/docs")


def _home(run: str) -> Path:
    """One Harness home by name, refusing anything that escapes the runs root."""
    home = (RUNS_ROOT / run).resolve()
    if not home.is_relative_to(RUNS_ROOT.resolve()) or not (home / "sessions").is_dir():
        raise HTTPException(404, f"no run {run!r} under {RUNS_ROOT}")
    return home


def _case_dir(case: str) -> Path | None:
    return case_dir_for(case, CASES_ROOT)


def _one(home: Path, pattern: str, wanted: str, run: str) -> Episode:
    """The single session a pattern matches, loaded and scored."""
    path = next((home / "sessions").glob(pattern), None)
    if path is None:
        raise HTTPException(404, f"no episode {wanted!r} in {run!r}")
    return load_episode(path, case_dir=_case_dir(case_of(path)))


#: One snapshot per home, replaced rather than accumulated: `{home: (stamp, episodes)}`.
#: An `lru_cache` keyed on the mtime grows instead of invalidating — while a sweep
#: is running every request is a new key, so eight successive copies of the same
#: 360-episode run stay resident (each `Episode` holds every step's reasoning and
#: result head, so that is gigabytes for one run).
_CACHE: dict[str, tuple[float, list[Episode]]] = {}
#: Held while a snapshot is built. The endpoints are sync `def`, so FastAPI runs
#: them on a threadpool and the two requests the UI fires together would
#: otherwise both load all 360 logs.
_LOADING = Lock()


def episodes_of(run: str) -> list[Episode]:
    """Every episode of a run, rebuilt only when something under it changed."""
    home = _home(run)
    # One stat per session, and the sort `find_sessions` pays for is not needed
    # to find a maximum.
    stamp = max((p.stat().st_mtime for p in session_paths(home)), default=0.0)
    key = str(home)
    with _LOADING:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        episodes = [
            load_episode(path, case_dir=_case_dir(case_of(path))) for path in find_sessions(home)
        ]
        _CACHE[key] = (stamp, episodes)
        return episodes


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    """Every Harness home under the runs root, biggest first."""
    homes = [d for d in RUNS_ROOT.glob("*") if (d / "sessions").is_dir()]
    # `session_paths`, not `find_sessions`: ordering them by mtime would stat
    # every one of a few hundred files per home just to count them.
    found = [{"run": home.name, "episodes": sum(1 for _ in session_paths(home))} for home in homes]
    return sorted(found, key=lambda r: -int(r["episodes"]))


@app.get("/api/runs/{run}/episodes")
def episodes(run: str) -> list[dict[str, Any]]:
    """One summary row per episode — what the list view shows."""
    return [episode.summary() for episode in episodes_of(run)]


@app.get("/api/runs/{run}/episodes/{session_id}")
def episode(run: str, session_id: str) -> dict[str, Any]:
    """One episode in full: every step's reasoning, call, and result head.

    Resolved by path rather than by scanning the run: the session id is the
    directory name, so this reads one log instead of all of them.
    """
    return _detail(_one(_home(run), f"*/{session_id}/{SESSION_FILE}", session_id, run))


@app.get("/api/runs/{run}/cases/{case}")
def case(run: str, case: str) -> dict[str, Any]:
    """The K samples of one case, for the sibling comparison.

    Globbed on the snapshot directory `case_of` reads the case out of, so this
    loads the group rather than the run.
    """
    home = _home(run)
    truth = _case_dir(case)
    group = [
        load_episode(path, case_dir=truth)
        for path in sorted(session_paths(home))
        if case_of(path) == case
    ]
    if not group:
        raise HTTPException(404, f"no episode on case {case!r} in {run!r}")
    return {"case": case, "stats": group_stats(group), "episodes": [_detail(e) for e in group]}


@app.get("/api/runs/{run}/report", response_class=PlainTextResponse)
def report(run: str) -> str:
    """The run as markdown, for the dashboard to render and for pasting into notes."""
    return render_report(run, episodes_of(run))


def _detail(episode: Episode) -> dict[str, Any]:
    """An episode as JSON. `Answer`'s frozensets become sorted lists.

    The three axes are what the graph comparison draws: `truth - found` is what
    the episode missed, `claimed - found` what it invented.
    """
    detail = episode.summary()
    detail["steps"] = [asdict(step) for step in episode.steps]
    detail["compactions"] = [
        {**asdict(pass_), "landed": pass_.landed} for pass_ in episode.compactions
    ]
    detail["submission"] = episode.submission
    if episode.answer is not None:
        detail["axes"] = {
            axis: {
                "found": sorted(episode.answer.found.get(axis, frozenset())),
                "missed": sorted(
                    episode.answer.truth.get(axis, frozenset())
                    - episode.answer.found.get(axis, frozenset())
                ),
                "invented": sorted(
                    episode.answer.claimed.get(axis, frozenset())
                    - episode.answer.found.get(axis, frozenset())
                ),
            }
            for axis in episode.answer.truth
        }
    return detail


def render_report(run: str, episodes: list[Episode]) -> str:
    """The numbers that decide whether this run is worth training on."""
    if not episodes:
        return f"# {run}\n\nNo episode ran.\n"
    stats = group_stats(episodes)
    finished = sum(1 for e in episodes if e.finish == "completed")
    failures = sum(1 for e in episodes for c in e.compactions if c.error)
    over = sum(1 for e in episodes if e.peak_input_tokens > COMPACTION_THRESHOLD)

    lines = [
        f"# {run}",
        "",
        f"{stats['cases']} case(s), {stats['episodes']} episode(s).",
        "",
        "## Outcome",
        "",
        "| | |",
        "|---|---|",
        f"| submission rate | {stats['submission_rate']:.3f} |",
        f"| ran to completion | {finished}/{len(episodes)} |",
        f"| pass@1 (mean of K) | {stats['pass_1']:.4f} |",
        f"| pass@k (best of K) | {stats['pass_k']:.4f} |",
        f"| cases with spread > 0 | {stats['varied_fraction']:.3f} |",
        "",
        "`spread > 0` is the share of an RLOO batch that can carry a gradient:",
        "a case whose K samples score alike contributes zero advantage to each of them.",
        "",
        "## Context",
        "",
        "| | |",
        "|---|---|",
        f"| compaction failures | {failures} |",
        f"| episodes past the {COMPACTION_THRESHOLD}-token threshold | {over}/{len(episodes)} |",
        "",
        "## Per case",
        "",
        "| case | k | pass@1 | pass@k | spread | submitted |",
        "|---|---|---|---|---|---|",
    ]
    for row in stats["per_case"]:
        lines.append(
            f"| {row['case']} | {row['k']} | {row['pass_1']:.3f} | {row['pass_k']:.3f} "
            f"| {row['spread']:.3f} | {row['submitted']}/{row['k']} |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["app", "render_report"]
