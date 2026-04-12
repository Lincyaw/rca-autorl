from __future__ import annotations

import ast
import json

from openai import AsyncOpenAI

from autorl.contracts import RuntimeContext, TaskOutcome, TaskSample, Trajectory
from autorl.rewards import reference_match_reward
from autorl.rewards.base import RewardStrategy


class SearchRewardStrategy(RewardStrategy):
    """Example reward strategy with optional LLM judge fallback."""

    async def compute(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
        outcome: TaskOutcome,
        runtime_context: RuntimeContext,
    ) -> float:
        reference = outcome.reference
        if reference is None and isinstance(sample.reference, dict):
            reference = sample.reference.get("answer")
        local_score = reference_match_reward(str(outcome.prediction or ""), reference)
        judge_base_url = runtime_context.metadata.get("judge_base_url")
        if not judge_base_url or local_score == 1.0 or not reference:
            return local_score

        judge_client = AsyncOpenAI(
            base_url=str(judge_base_url),
            api_key="EMPTY",
            http_client=runtime_context.http_client,
            max_retries=0,
        )
        judge_prompt = (
            "You are an evaluation assistant. Determine whether pred_answer is equivalent "
            "to ground truth answer. Output JSON with rationale and judgement (correct/incorrect).\n\n"
            f"question: {trajectory.metadata.get('question', '')}\n"
            f"ground truth answers: {reference}\n"
            f"pred_answer: {str(outcome.prediction)[:200]}\n\n"
            "Your output:"
        )
        try:
            judge_completion = await judge_client.chat.completions.create(
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=1.0,
                max_completion_tokens=2048,
                store=False,
            )
        except Exception:
            return local_score
        judge_response = judge_completion.choices[0].message.content or ""
        return _parse_judge_result(judge_response)


def _parse_judge_result(raw_response: str) -> float:
    parsed = None
    for parse_fn in (json.loads, ast.literal_eval):
        try:
            candidate = raw_response.split("```json")[-1].split("```")[0].strip()
            parsed = parse_fn(candidate)
            break
        except Exception:
            continue
    if parsed is None and '"judgement": "incorrect"' in raw_response:
        parsed = {"judgement": "incorrect"}
    if parsed is None and '"judgement": "correct"' in raw_response:
        parsed = {"judgement": "correct"}
    if not isinstance(parsed, dict):
        return 0.0
    return float(parsed.get("judgement") == "correct")
