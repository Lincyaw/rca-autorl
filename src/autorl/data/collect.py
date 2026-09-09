"""Run RCA cases through DeepSeek Harness against a teacher model, for SFT.

A rollout gets its model from AReaL's proxy; distillation has no proxy, so this
entrypoint aims the same composition at a teacher endpoint. Same profile, same
bundle, same scenario layer, same model route — `autorl.harness.model_route`
composes it for both, and only the endpoint differs. Every episode writes an
ordinary `dsh` session under one Harness home, which is exactly what
`autorl.data.export` already reads.

    export RCA_GATEWAY_BASE_URL=... RCA_GATEWAY_API_KEY=...
    python -m autorl.data.collect <manifest.jsonl> <dsh-home> --limit 10
    python -m autorl.data.export <dsh-home> <out.jsonl>

With no `--base-url` the episode runs on `sdk-minimal`'s own
`deepseek-official` route, which reads `DEEPSEEK_API_KEY` and serves only that
catalog's model ids.

The session id carries the case id, so an exported row's `sample_id` traces
back to the manifest without a second sidecar file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from autorl.agent import resolve_data_dir, submitted_result
from autorl.data.samples import load_manifest_samples
from autorl.fpg import parse_submission
from autorl.harness import ModelRoute, model_route, require_bundle, run_episode
from autorl.interfaces import RCASample


def _first(sample: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """The first key present with a value, treating 0 and "" as present."""
    for key in keys:
        value = sample.get(key)
        if value is not None and value != "":
            return value
    return None


@dataclass
class Episode:
    """What one collected case is worth knowing about without reading its log."""

    case_id: str
    session_id: str
    finish_reason: str
    submitted: bool
    in_contract: bool = False
    error: str = ""


def collect_case(
    sample: dict[str, Any],
    *,
    dsh_home: Path,
    route: ModelRoute,
    max_tokens: int,
    timeout: float,
    dataset_root: str,
) -> Episode:
    # `or` would skip a row whose id is 0, and the corpus indexes from zero.
    case_id = str(_first(sample, ("id", "source", "datapack_name")))
    incident = str(sample.get("question") or sample.get("incident") or "").strip()
    if not incident:
        raise ValueError(f"case {case_id} has no question/incident")
    session_id = f"case{case_id}-{uuid.uuid4().hex[:8]}"
    try:
        result = run_episode(
            dsh_home=dsh_home,
            route=route,
            # A manifest row is untyped JSON until it is read as one.
            cwd=resolve_data_dir(cast(RCASample, sample), dataset_root),
            prompt=incident,
            session_id=session_id,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as error:  # one failed case must not lose the rest of the batch
        return Episode(
            case_id, session_id, "error", False, error=f"{type(error).__name__}: {error}"
        )
    submission = submitted_result(result)
    episode = Episode(case_id, result.session_id, str(result.finish_reason), submission is not None)
    if submission is None:
        return episode
    # The harness tool validated these arguments before it accepted the call, so
    # this can only fail if the two sides of the contract have drifted apart —
    # which is worth one parse per episode to learn now rather than at training
    # time, when the row is already in the dataset.
    try:
        parse_submission(submission)
    except ValidationError as error:
        episode.error = f"submission is out of contract: {error.error_count()} violation(s)"
        return episode
    episode.in_contract = True
    return episode


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m autorl.data.collect")
    parser.add_argument("manifest", help="RCA manifest .jsonl/.json")
    parser.add_argument("dsh_home", help="Harness home to write sessions into")
    parser.add_argument("--limit", type=int, default=0, help="first N cases (0 = all)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default="DeepSeek-V4-pro")
    parser.add_argument("--base-url", default=os.getenv("RCA_GATEWAY_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("RCA_GATEWAY_API_KEY", ""))
    parser.add_argument("--scenario", default="rca")
    parser.add_argument("--concurrency", type=int, default=1)
    # Mirror the limits the student is actually served under, so the teacher
    # trajectory is one the student could have produced: `gconfig.max_new_tokens`
    # and `sglang.context_length` from configs/train/dsh_rca_smoke.yaml — these
    # two defaults move only when that file does. The window is also what
    # compaction triggers below, which is what keeps an exported row inside the
    # SFT config's max_length.
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=None, help="unset: endpoint default")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--report", default="", help="write the per-case summary as JSONL")
    args = parser.parse_args(argv)

    dsh_home = Path(args.dsh_home).expanduser().resolve()
    require_bundle(dsh_home)
    samples = load_manifest_samples(args.manifest)[args.offset :]
    if args.limit:
        samples = samples[: args.limit]
    dataset_root = args.dataset_root or str(Path(args.manifest).expanduser().resolve().parent)
    route = model_route(
        scenario=args.scenario,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        context_window=args.context_window,
        temperature=args.temperature,
    )
    print(f"teacher: provider={route.provider} model={route.model} {args.base_url}", flush=True)

    def run(sample: dict[str, Any]) -> Episode:
        episode = collect_case(
            sample,
            dsh_home=dsh_home,
            route=route,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            dataset_root=dataset_root,
        )
        print(
            f"{episode.case_id}\t{episode.finish_reason}\t"
            f"submitted={episode.submitted}\tin_contract={episode.in_contract}\t{episode.error}",
            flush=True,
        )
        return episode

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        episodes = list(pool.map(run, samples))

    submitted = sum(1 for e in episodes if e.submitted)
    in_contract = sum(1 for e in episodes if e.in_contract)
    print(
        f"collected {len(episodes)} episode(s), {submitted} with a submission "
        f"({in_contract} in contract), into {dsh_home}"
    )
    if args.report:
        report = Path(args.report).expanduser()
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode.__dict__, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
