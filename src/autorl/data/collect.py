"""Run RCA cases through DeepSeek Harness against a teacher model, for SFT.

The RL path gets its model from AReaL's rollout proxy; SFT distillation has no
proxy, so this entrypoint points the same harness composition — `sdk-minimal`
plus the `agent/rca-harness` bundle and the scenario patch — at a teacher
endpoint instead. Every episode writes an ordinary `dsh` session under one
Harness home, which is exactly what `autorl.data.export` already reads.

    export RCA_TEACHER_BASE_URL=... RCA_TEACHER_API_KEY=...
    python -m autorl.data.collect <manifest.jsonl> <dsh-home> --limit 10
    python -m autorl.data.export <dsh-home> <out.jsonl>

A gateway is reached as a *route*, not as a `base_url` override: `--base-url`
layers `agent/profiles/openai-gateway.patch.yml`, which mounts the shipped
`llm-pi-ai` adapter and declares one `openai-completions` route from the same
environment variables. That layer's header says why the base_url override is
not enough. With no `--base-url` the episode runs on `sdk-minimal`'s own
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
from typing import Any

from deepseek_harness import DeepSeekHarness

from autorl.agent import resolve_data_dir, submitted_result
from autorl.data.samples import load_manifest_samples
from autorl.harness import PROFILE, require_bundle, scenario_patch

GATEWAY_SCENARIO = "openai-gateway"
GATEWAY_ROUTE = "gateway"
DEEPSEEK_ROUTE = "deepseek-official"


@dataclass
class Episode:
    """What one collected case is worth knowing about without reading its log."""

    case_id: str
    session_id: str
    finish_reason: str
    submitted: bool
    error: str = ""


@dataclass
class Teacher:
    """The endpoint an episode runs against, in the form the harness takes it."""

    model: str
    provider: str
    patches: tuple[str, ...]
    env: dict[str, str]


def teacher(scenario: str, model: str, base_url: str, api_key: str, context_window: int) -> Teacher:
    """Compose the model route: a declared gateway route, or the shipped one."""
    env = {"DSH_CONTEXT_WINDOW": str(context_window)}
    patches = (scenario_patch(scenario),)
    if not base_url:
        return Teacher(model, DEEPSEEK_ROUTE, patches, env)
    return Teacher(
        model,
        GATEWAY_ROUTE,
        (*patches, scenario_patch(GATEWAY_SCENARIO)),
        # The route reads these; the patch holds no literal of its own.
        {
            **env,
            "RCA_TEACHER_BASE_URL": base_url,
            "RCA_TEACHER_API_KEY": api_key,
            "RCA_TEACHER_MODEL": model,
        },
    )


def collect_case(
    sample: dict[str, Any],
    *,
    dsh_home: Path,
    teacher: Teacher,
    max_tokens: int,
    timeout: float,
    dataset_root: str,
) -> Episode:
    case_id = str(sample.get("id") or sample.get("source") or sample.get("datapack_name"))
    incident = str(sample.get("question") or sample.get("incident") or "").strip()
    if not incident:
        raise ValueError(f"case {case_id} has no question/incident")
    session_id = f"case{case_id}-{uuid.uuid4().hex[:8]}"
    try:
        with DeepSeekHarness(
            dsh_home=str(dsh_home),
            profile=PROFILE,
            patches=teacher.patches,
            provider=teacher.provider,
            cwd=resolve_data_dir(sample, dataset_root),  # type: ignore[arg-type]
            model=teacher.model,
            max_tokens=max_tokens,
            env=teacher.env,
            request_timeout_seconds=timeout,
        ) as harness:
            result = harness.run(incident, session_id=session_id)
    except Exception as error:  # one failed case must not lose the rest of the batch
        return Episode(case_id, session_id, "error", False, f"{type(error).__name__}: {error}")
    return Episode(
        case_id, result.session_id, str(result.finish_reason), submitted_result(result) is not None
    )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m autorl.data.collect")
    parser.add_argument("manifest", help="RCA manifest .jsonl/.json")
    parser.add_argument("dsh_home", help="Harness home to write sessions into")
    parser.add_argument("--limit", type=int, default=0, help="first N cases (0 = all)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default="DeepSeek-V4-pro")
    parser.add_argument("--base-url", default=os.getenv("RCA_TEACHER_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("RCA_TEACHER_API_KEY", ""))
    parser.add_argument("--scenario", default="rca")
    parser.add_argument("--concurrency", type=int, default=1)
    # Mirror the limits the student is actually served under, so the teacher
    # trajectory is one the student could have produced: `gconfig.max_new_tokens`
    # and `sglang.context_length` from configs/train/dsh_rca_smoke.yaml. The
    # window is also what compaction triggers below, which is what keeps an
    # exported row inside the SFT config's max_length.
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--context-window", type=int, default=32768)
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
    route = teacher(args.scenario, args.model, args.base_url, args.api_key, args.context_window)
    print(f"teacher: provider={route.provider} model={route.model} {args.base_url}", flush=True)

    def run(sample: dict[str, Any]) -> Episode:
        episode = collect_case(
            sample,
            dsh_home=dsh_home,
            teacher=route,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            dataset_root=dataset_root,
        )
        print(
            f"{episode.case_id}\t{episode.finish_reason}\t"
            f"submitted={episode.submitted}\t{episode.error}",
            flush=True,
        )
        return episode

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        episodes = list(pool.map(run, samples))

    submitted = sum(1 for e in episodes if e.submitted)
    print(f"collected {len(episodes)} episode(s), {submitted} with a submission, into {dsh_home}")
    if args.report:
        report = Path(args.report).expanduser()
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode.__dict__, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
