"""Prepare `datapacks/ops-lite` for training: the incident manifest and the ground truth.

The released corpus is 500 cases of telemetry plus annotation. Two things it
does not ship are what a run needs, and both are derived rather than authored:

- **The incident.** A case has no prompt. What it has is `conclusion.parquet`,
  the per-endpoint normal-vs-abnormal comparison whose non-empty `Issues` column
  is exactly the set of SLO violations an on-call engineer would be paged for.
  The manifest reproduces the wording the earlier train-ticket manifest used,
  and on the 200 cases the two corpora share it reproduces the endpoint list
  exactly.
- **A ground truth the bound models accept.** `causal_graph_verified.json` is an
  `fpg.Scenario` already, but it declares `microservices-0.4.0` while
  `configs/fpg/microservices.toml` is 0.5.0, and the factory pins the version
  string exactly. 0.5.0 only adds — three predicates, one mechanism, the `link`
  entity type — so restamping is the additive migration fpg's policy describes,
  not a reinterpretation. Three cases are also missing `testbed`, which the
  release manifest has under `system`.

The trainable cases are split once, by a hash of the case name, into
`train.jsonl` and `eval.jsonl`. The method spec evaluates on a fixed held-out
set chosen before training, and a hash keeps the split stable under re-runs and
under corpus additions. Both steps are idempotent, and `--check` reports without
writing.

    python -m autorl.dataset datapacks/ops-lite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from autorl.fpg import schema

INCIDENT_TEMPLATE = (
    "The following API endpoints are experiencing possible SLO violations "
    "and need investigation:\n{endpoints}\n\n"
    "Please investigate the root cause of these SLO violations."
)


@dataclass
class Report:
    """What one preparation pass changed, and what it could not."""

    cases: int = 0
    restamped: int = 0
    testbed_filled: int = 0
    incidents: int = 0
    held_out: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)


def slo_violations(case_dir: Path) -> list[str]:
    """Endpoints whose `Issues` column is non-empty, in the corpus's own order.

    Read with pyarrow rather than DuckDB: `conclusion.parquet` is one small
    table per case and AReaL already brings pyarrow, so preparing the corpus
    needs no dependency the training run does not already have.
    """
    import pyarrow.parquet as pq

    conclusion = case_dir / "conclusion.parquet"
    if not conclusion.is_file():
        return []
    table = pq.read_table(conclusion, columns=["SpanName", "Issues"])  # type: ignore[no-untyped-call]
    names = table.column("SpanName").to_pylist()
    issues = table.column("Issues").to_pylist()
    return [
        str(name)
        for name, issue in zip(names, issues, strict=True)
        if issue not in ("{}", "", None)
    ]


def incident(endpoints: list[str]) -> str:
    return INCIDENT_TEMPLATE.format(endpoints="\n".join(f"- {name}" for name in endpoints))


def normalize_ground_truth(
    path: Path, *, testbed: str, vocab_version: str, write: bool
) -> tuple[bool, bool]:
    """Restamp one verified graph. Returns (restamped, testbed filled)."""
    scenario: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    restamped = scenario.get("vocab_version") != vocab_version
    filled = not scenario.get("testbed")
    if restamped:
        scenario["vocab_version"] = vocab_version
    if filled:
        scenario["testbed"] = testbed
    if write and (restamped or filled):
        path.write_text(json.dumps(scenario, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return restamped, filled


EVAL_SHARE = 0.1


def is_held_out(name: str) -> bool:
    """One case in ten, chosen by its name alone so the split never moves."""
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100 < EVAL_SHARE * 100


def prepare(root: Path, *, write: bool = True) -> Report:
    """Restamp every ground truth, and write the manifests of trainable cases.

    Trainable means both halves are there: a symptom the episode can start from,
    and a graph it can be scored against, and the annotation side did not flag
    the case. Anything else stays in the corpus and goes into
    `data.excluded.jsonl` with the reason — 51 cases show no SLO violation at all
    (the injected fault never reached an endpoint; 32 of the 38 otel-demo cases
    are like this), 3 carry an empty verified graph, and 15 are marked
    `.invalid`.
    """
    bundle = schema()
    manifest = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = Report()
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(manifest):
        name = str(entry["name"])
        case_dir = root / "cases" / name
        if not case_dir.is_dir():
            report.skipped.append((name, "no case directory"))
            continue
        report.cases += 1

        reasons: list[str] = []
        verified = case_dir / "causal_graph_verified.json"
        if not verified.is_file():
            reasons.append("no causal_graph_verified.json")
        else:
            restamped, filled = normalize_ground_truth(
                verified,
                testbed=str(entry["system"]),
                vocab_version=bundle.profile.vocab_version,
                write=write,
            )
            report.restamped += restamped
            report.testbed_filled += filled
            try:
                bundle.Scenario.model_validate_json(verified.read_text(encoding="utf-8"))
            except ValidationError as error:
                reasons.append(f"ground truth invalid: {error.errors()[0]['msg']}")

        endpoints = slo_violations(case_dir)
        if not endpoints:
            reasons.append("no SLO violation in conclusion.parquet")
        # An empty marker file the annotation pipeline leaves on a case it
        # rejected. What exactly it rejected is not recorded, and the graphs
        # under it look ordinary, so this is a precaution rather than a
        # diagnosis — drop the line to train on them.
        if (case_dir / ".invalid").exists():
            reasons.append("marked .invalid by the annotation pipeline")

        if reasons:
            report.skipped.append((name, "; ".join(reasons)))
            continue
        report.incidents += 1
        report.held_out += is_held_out(name)
        rows.append(
            {
                "id": index,
                "source": name,
                "datapack_name": name,
                "system": entry["system"],
                "question": incident(endpoints),
                # The released manifest's own summary of the answer, kept for
                # stratification and quick sanity checks. The scored ground
                # truth is the verified graph, not this.
                "root_services": entry.get("root_services", []),
                "primary_kind": entry.get("primary_kind"),
                "chaos_family": entry.get("chaos_family"),
                "longest_path": entry.get("longest_path"),
            }
        )

    if write:
        for split, held_out in (("train", False), ("eval", True)):
            with (root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    if is_held_out(str(row["source"])) == held_out:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Every case the manifest lists but neither split does, with the
        # reason. An excluded case is still a case — keeping the ledger beside
        # the manifest is what makes "446 of 500" auditable rather than folklore.
        with (root / "data.excluded.jsonl").open("w", encoding="utf-8") as handle:
            for name, reason in report.skipped:
                handle.write(json.dumps({"source": name, "reason": reason}) + "\n")
    return report


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m autorl.dataset")
    parser.add_argument("root", nargs="?", default="datapacks/ops-lite")
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    report = prepare(root, write=not args.check)
    print(
        f"{report.cases} case(s): {report.incidents} incident(s) "
        f"({report.held_out} held out), "
        f"{report.restamped} ground truth restamped, {report.testbed_filled} testbed filled"
    )
    for name, reason in report.skipped[:20]:
        print(f"  skipped {name}: {reason}")
    if len(report.skipped) > 20:
        print(f"  ... and {len(report.skipped) - 20} more")


if __name__ == "__main__":
    main(sys.argv[1:])
