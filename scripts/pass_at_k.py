"""Sample K episodes per held-out case and report pass@1 against pass@k.

The question this answers is whether RL has anything to optimise. RLOO centres a
trajectory against its siblings (method spec §3.2), so if K samples of a case
never disagree the advantage is zero for every one of them and the whole batch
is wasted. pass@k above pass@1 is the same statement in outcome terms: some
sample found something the average one did not, and a policy gradient can move
mass toward it.

Runs `autorl.data.collect`'s episode path unchanged — same profile, same bundle,
same route — against one or more served checkpoints, then scores each episode
with the reward the trainer uses (`difficulty.weighted_score` on flat weights,
which is the three-axis graph F1 of method spec §2).

    ./scripts/serve_sft.sh &                     # or several, on several ports
    python -m scripts.pass_at_k datapacks/ops-lite/eval.jsonl .runs/passk \
        --samples 8 --base-url http://127.0.0.1:30200/v1 \
                    --base-url http://127.0.0.1:30201/v1

Reports, per case: how many of the K samples submitted at all, the mean score
(pass@1 in expectation) and the max (pass@k), and the spread between them. A
case whose K scores are identical contributes nothing to an RLOO batch, so the
fraction of cases with non-zero spread is the number to read.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autorl.agent import resolve_data_dir
from autorl.data.samples import read_jsonl
from autorl.difficulty import weighted_score
from autorl.harness import model_route, require_bundle, run_episode
from autorl.reward import answer_of, submitted_result, truth_for_case


@dataclass
class Sample:
    """One episode's outcome, scored the way the trainer scores it."""

    case_id: str
    session_id: str
    score: float = 0.0
    submitted: bool = False
    finish: str = ""
    error: str = ""


@dataclass
class CaseResult:
    """The K samples of one case, and what they say about within-group variance."""

    case_id: str
    source: str
    samples: list[Sample] = field(default_factory=list)

    @property
    def scores(self) -> list[float]:
        return [s.score for s in self.samples]

    @property
    def pass_1(self) -> float:
        """The expected score of one sample — what a single rollout is worth."""
        return statistics.fmean(self.scores) if self.samples else 0.0

    @property
    def pass_k(self) -> float:
        """The best of K — what the group's ceiling is."""
        return max(self.scores) if self.samples else 0.0

    @property
    def spread(self) -> float:
        return self.pass_k - min(self.scores) if self.samples else 0.0


def run_sample(
    sample: dict[str, Any],
    *,
    dsh_home: Path,
    base_url: str,
    api_key: str,
    model: str,
    scenario: str,
    context_window: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
    dataset_root: str,
) -> Sample:
    case_id = str(sample.get("id"))
    session_id = f"case{case_id}-{uuid.uuid4().hex[:8]}"
    route = model_route(
        scenario=scenario,
        model=model,
        base_url=base_url,
        api_key=api_key,
        context_window=context_window,
        temperature=temperature,
    )
    case_dir = resolve_data_dir(sample, dataset_root)
    try:
        result = run_episode(
            dsh_home=dsh_home,
            route=route,
            cwd=case_dir,
            prompt=str(sample["question"]).strip(),
            session_id=session_id,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as error:  # one lost episode must not lose the group
        return Sample(case_id, session_id, error=f"{type(error).__name__}: {error}")

    submission = submitted_result(result.events)
    out = Sample(
        case_id,
        result.session_id,
        submitted=submission is not None,
        finish=str(result.finish_reason),
    )
    if submission is None:
        return out
    # Flat weights: the difficulty weighting of §3.1 is defined against a
    # sibling group's own hit rates, and reading pass@k through it would make
    # the number depend on the group rather than on the answer.
    out.score = weighted_score(answer_of(submission, truth_for_case(Path(case_dir))), {})
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="held-out manifest, e.g. datapacks/ops-lite/eval.jsonl")
    parser.add_argument("dsh_home", help="Harness home to write sessions into")
    parser.add_argument("--samples", type=int, default=8, help="K")
    parser.add_argument("--limit", type=int, default=0, help="first N cases (0 = all)")
    parser.add_argument(
        "--base-url",
        action="append",
        default=[],
        help="a served endpoint including /v1; repeat to spread the load over several",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="rca-sft")
    parser.add_argument("--scenario", default="rca")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--context-window", type=int, default=32768)
    # The prompt at the compaction threshold overshoots it — a request already in
    # flight when the meter trips still carries the pre-compaction surface, and
    # the largest observed was 21807 tokens against a nominal 19661. 21807 +
    # 10240 = 32047 fits 32768; the 12288 the trainer uses does not, and sglang
    # answers a request that does not fit with a bare 400 that ends the episode.
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--dataset-root", default="datapacks/ops-lite/cases")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--report", default="", help="write per-sample rows as JSONL")
    args = parser.parse_args(argv)

    endpoints = args.base_url or ["http://127.0.0.1:30111/v1"]
    cases = list(read_jsonl(Path(args.manifest)))
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit(f"{args.manifest} has no cases")

    dsh_home = Path(args.dsh_home).expanduser().resolve()
    dsh_home.mkdir(parents=True, exist_ok=True)
    require_bundle(dsh_home)

    # One flat work list of (case, replica) so every endpoint stays busy: a
    # per-case barrier would leave three servers idle while the longest episode
    # of a group finishes.
    work = [(case, i) for case in cases for i in range(args.samples)]
    print(f"{len(cases)} case(s) x {args.samples} sample(s) over {len(endpoints)} endpoint(s)")

    def one(item: tuple[dict[str, Any], int]) -> Sample:
        case, index = item
        return run_sample(
            case,
            dsh_home=dsh_home,
            base_url=endpoints[index % len(endpoints)],
            api_key=args.api_key,
            model=args.model,
            scenario=args.scenario,
            context_window=args.context_window,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            dataset_root=args.dataset_root,
        )

    results: dict[str, CaseResult] = {
        str(case["id"]): CaseResult(str(case["id"]), str(case.get("source", ""))) for case in cases
    }
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for done, sample in enumerate(pool.map(one, work), 1):
            results[sample.case_id].samples.append(sample)
            flag = f"{sample.score:.3f}" if sample.submitted else (sample.error or "no submission")
            print(f"  [{done}/{len(work)}] case {sample.case_id}: {flag}", flush=True)

    if args.report:
        with Path(args.report).open("w", encoding="utf-8") as handle:
            for case in results.values():
                for sample in case.samples:
                    handle.write(
                        json.dumps(
                            {
                                "case_id": sample.case_id,
                                "source": case.source,
                                "session_id": sample.session_id,
                                "score": sample.score,
                                "submitted": sample.submitted,
                                "finish": sample.finish,
                                "error": sample.error,
                            }
                        )
                        + "\n"
                    )

    report(list(results.values()), args.samples)
    return 0


def report(cases: list[CaseResult], k: int) -> None:
    scored = [c for c in cases if c.samples]
    if not scored:
        print("no episode ran")
        return
    submitted = sum(s.submitted for c in scored for s in c.samples)
    total = sum(len(c.samples) for c in scored)
    varied = [c for c in scored if c.spread > 0]

    print()
    for case in sorted(scored, key=lambda c: -c.spread):
        marks = " ".join(f"{s.score:.2f}" if s.submitted else "  - " for s in case.samples)
        print(f"case {case.case_id:>4}  p@1 {case.pass_1:.3f}  p@{k} {case.pass_k:.3f}  [{marks}]")

    print()
    print(f"submission rate       {submitted}/{total} = {submitted / total:.3f}")
    print(f"pass@1  (mean of K)   {statistics.fmean(c.pass_1 for c in scored):.4f}")
    print(f"pass@{k} (best of K)   {statistics.fmean(c.pass_k for c in scored):.4f}")
    print(f"gain                  {statistics.fmean(c.pass_k - c.pass_1 for c in scored):+.4f}")
    print(
        f"cases with spread > 0 {len(varied)}/{len(scored)} = {len(varied) / len(scored):.3f}"
        "  <- the fraction an RLOO batch can learn from"
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
