from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from autorl.contracts import AgentInput, TaskOutcome, TaskSample, Trajectory

from .base import TaskAdapter

_ID_KEYS = ("sample_id", "qid", "query_id", "id")
_RESERVED_KEYS = {"question", "answer", "sample_id", "qid", "query_id", "id", "messages"}


class SearchTaskAdapter(TaskAdapter):
    """Example task adapter for QA-style search-agent samples."""

    task_type = "search"

    def validate_sample(self, raw_sample: Mapping[str, Any]) -> TaskSample:
        question = raw_sample.get("question")
        if question is None:
            raise ValueError("search sample missing required key: question")

        sample_id = ""
        for key in _ID_KEYS:
            value = raw_sample.get(key)
            if value is not None and str(value).strip():
                sample_id = str(value)
                break
        if not sample_id:
            sample_id = uuid4().hex

        answer = raw_sample.get("answer")
        target = {"answer": answer} if answer is not None else None

        metadata: dict[str, Any] = {}
        for key, value in raw_sample.items():
            if key not in _RESERVED_KEYS:
                metadata[key] = value

        model_input: dict[str, Any] = {"question": str(question)}
        if isinstance(raw_sample.get("messages"), list):
            model_input["messages"] = list(raw_sample["messages"])

        return TaskSample(
            sample_id=sample_id,
            task_type=self.task_type,
            input=model_input,
            target=target,
            reference=target,
            metadata=metadata,
            raw=dict(raw_sample),
        )

    def to_agent_input(self, sample: TaskSample) -> AgentInput:
        question = str(sample.input.get("question", ""))
        messages_raw = sample.input.get("messages")
        if isinstance(messages_raw, list) and messages_raw:
            messages = list(messages_raw)
        else:
            messages = [{"role": "user", "content": question}]

        return AgentInput(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            instruction="Answer the question with external evidence when needed.",
            context={"question": question},
            messages=messages,
            available_tools=[
                {"name": "search"},
                {"name": "visit"},
            ],
            constraints={},
            metadata=dict(sample.metadata),
            raw_sample=dict(sample.raw),
        )

    def to_task_outcome(self, sample: TaskSample, trajectory: Trajectory) -> TaskOutcome:
        prediction = _extract_prediction(trajectory.final_output)
        reference = None
        if sample.reference:
            reference = sample.reference.get("answer")

        success = None
        if isinstance(reference, str) and reference.strip():
            pred_norm = str(prediction).strip().lower()
            ref_norm = reference.strip().lower()
            success = bool(pred_norm and (pred_norm == ref_norm or pred_norm in ref_norm or ref_norm in pred_norm))

        return TaskOutcome(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            prediction=prediction,
            reference=reference,
            success=success,
            termination_reason=str(trajectory.final_output.get("termination", trajectory.status.value)),
            metrics=dict(trajectory.summary_stats),
            metadata=dict(sample.metadata),
        )


def _extract_prediction(final_output: Mapping[str, Any] | Any) -> Any:
    if isinstance(final_output, Mapping):
        for key in ("prediction", "answer", "output", "final_output"):
            if key in final_output:
                return final_output[key]
    return final_output
