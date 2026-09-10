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
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from autorl.trajectory import Episode, case_of, find_sessions, group_stats, load_episode

#: Where to look for Harness homes. A home is any directory with `sessions/`.
RUNS_ROOT = Path(os.getenv("RCA_RUNS_ROOT", ".runs")).expanduser()
#: Where the corpus cases live, for the ground truth a score needs.
CASES_ROOT = Path(os.getenv("RCA_DATASET_ROOT", "datapacks/ops-lite/cases")).expanduser()
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
    directory = CASES_ROOT / case
    return directory if directory.is_dir() else None


def _load(path: Path) -> Episode:
    """One episode, scored when its case is still on this machine."""
    return load_episode(path, case_dir=_case_dir(case_of(path)))


@lru_cache(maxsize=8)
def _episodes(home: str, _stamp: float) -> list[Episode]:
    """Every episode of a run. Keyed by mtime so a finished run invalidates once."""
    return [_load(path) for path in find_sessions(Path(home))]


def episodes_of(run: str) -> list[Episode]:
    home = _home(run)
    newest = max((p.stat().st_mtime for p in find_sessions(home)), default=0.0)
    return _episodes(str(home), newest)


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    """Every Harness home under the runs root, newest first."""
    homes = [d for d in RUNS_ROOT.glob("*") if (d / "sessions").is_dir()]
    found = [{"run": home.name, "episodes": sum(1 for _ in find_sessions(home))} for home in homes]
    return sorted(found, key=lambda r: -int(r["episodes"]))


@app.get("/api/runs/{run}/episodes")
def episodes(run: str) -> list[dict[str, Any]]:
    """One summary row per episode — what the list view shows."""
    return [episode.summary() for episode in episodes_of(run)]


@app.get("/api/runs/{run}/episodes/{session_id}")
def episode(run: str, session_id: str) -> dict[str, Any]:
    """One episode in full: every step's reasoning, call, and result head."""
    for found in episodes_of(run):
        if found.session_id == session_id:
            return _detail(found)
    raise HTTPException(404, f"no episode {session_id!r} in {run!r}")


@app.get("/api/runs/{run}/cases/{case}")
def case(run: str, case: str) -> dict[str, Any]:
    """The K samples of one case, for the sibling comparison."""
    group = [e for e in episodes_of(run) if e.case == case]
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
