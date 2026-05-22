from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import json5
from llmharness.audit.extractor.state import ExtractionState
from llmharness.audit.extractor.tools import build_extractor_tools
from llmharness.schema import Event
from openai import AsyncOpenAI

from agentm.core.abi import ToolResult, ToolTerminate
from autorl.contracts import (
    AgentInput,
    AgentRunResult,
    RuntimeContext,
    TaskOutcome,
    TaskSample,
    TokenUsage,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
    TrajectoryStepType,
)
from autorl.rewards.base import RewardStrategy
from autorl.runtime.base import AgentRuntime
from autorl.tasks.base import TaskAdapter

_TOOL_STOP_TOKENS = ["\n<tool_response>", "<tool_response>"]
_GRAPH_EDIT_OPS = {
    "add_node",
    "update_node",
    "delete_node",
    "add_edge",
    "update_edge",
    "delete_edge",
}

GRAPH_EDIT_RL_HINT = """You are still the llmharness cognitive-audit extractor.
Use the same event schema and reference rules as before, but the runtime now
executes the real AgentM extractor tools.

For RL, the first tool call MUST be an incremental graph_edit. Do not call
submit_events_batch until at least one graph_edit call succeeds:
- graph_edit({"op":"add_node","node":{...}})
- graph_edit({"op":"update_node","node_id":1,"node":{...}})
- graph_edit({"op":"delete_node","node_id":1})
- graph_edit({"op":"add_edge","edge":{"src":1,"dst":2,"kind":"data","reason":"...","cited_entities":["..."]}})
- graph_edit({"op":"update_edge","edge_selector":{"src":1,"dst":2},"edge":{...}})
- graph_edit({"op":"delete_edge","edge_selector":{"src":1,"dst":2}})

After graph_edit calls, finalize with:
submit_events_batch({"events":[],"done":true})

If you cannot use graph_edit yet, the legacy submit_events({"events":[...]})
format is still accepted and will be translated to AgentM submit_events_batch.

Return exactly one JSON object inside <tool_call></tool_call> per turn.
"""


class OpsLiteTaskAdapter(TaskAdapter):
    """Adapter for ops-lite llmharness/distill rows used by extractor RL."""

    task_type = "ops_lite_tool_call"

    def validate_sample(self, raw_sample: Mapping[str, Any]) -> TaskSample:
        raw = dict(raw_sample)
        sample_input = raw.get("input") or {}
        if not isinstance(sample_input, Mapping):
            raise ValueError("ops-lite row requires input object")
        system = str(sample_input.get("system", "")).rstrip()
        user = str(sample_input.get("user", "")).rstrip()
        if not user:
            raise ValueError("ops-lite row requires input.user")
        target = raw.get("target") or {}
        sample_id = str(raw.get("id") or raw.get("sample_id") or uuid4().hex)
        return TaskSample(
            sample_id=sample_id,
            task_type=self.task_type,
            input={"system": system, "user": user},
            target=target,
            reference=_target_reference(target),
            metadata={"target_tool_name": "submit_events_batch"},
            raw=raw,
        )

    def to_agent_input(self, sample: TaskSample) -> AgentInput:
        original_system = str(sample.input.get("system", "")).rstrip()
        system = GRAPH_EDIT_RL_HINT
        if original_system:
            system += "\nOriginal extractor instructions:\n" + original_system
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(sample.input.get("user", "")).rstrip()},
        ]
        return AgentInput(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            instruction="Edit and submit the extractor graph with AgentM tools.",
            context=dict(sample.input),
            messages=messages,
            metadata=dict(sample.metadata),
            raw_sample=dict(sample.raw),
        )

    def to_task_outcome(self, sample: TaskSample, trajectory: Trajectory) -> TaskOutcome:
        metrics = dict(trajectory.summary_stats)
        expected_count = float((sample.reference or {}).get("event_count") or 0)
        actual_count = float(metrics.get("event_count", 0.0))
        if expected_count > 0:
            metrics["event_count_match"] = 1.0 if actual_count == expected_count else 0.0
        success = bool(metrics.get("committed", 0.0) and metrics.get("tool_ok", 0.0))
        return TaskOutcome(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            prediction=trajectory.final_output.get("tool_payload")
            or trajectory.final_output.get("prediction"),
            reference=sample.reference,
            success=success,
            termination_reason=str(
                trajectory.final_output.get("termination", trajectory.status.value)
            ),
            metrics=metrics,
            metadata=dict(sample.metadata),
        )


class OpsLiteToolRuntime(AgentRuntime):
    """Multi-turn rollout runtime executing real AgentM extractor tools."""

    def __init__(self, temperature: float = 0.0) -> None:
        self.temperature = temperature

    async def run(
        self,
        agent_input: AgentInput,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        if not runtime_context.model_endpoint:
            raise ValueError("OpsLiteToolRuntime requires RuntimeContext.model_endpoint")

        max_tokens = int(runtime_context.metadata.get("max_tokens_per_turn", 1200))
        max_calls = max(1, int(runtime_context.limits.max_llm_calls))
        input_payload = _input_payload(agent_input)
        state = _state_from_payload(input_payload)
        tools = {tool.name: tool for tool in build_extractor_tools(state)}
        messages = list(agent_input.messages)
        steps: list[TrajectoryStep] = []
        completion_ids: list[str] = []
        tool_payloads: list[dict[str, Any]] = []
        op_counts = {op: 0 for op in _GRAPH_EDIT_OPS}
        stats: dict[str, float] = {
            "has_tool_call": 0.0,
            "valid_json": 0.0,
            "tool_ok": 0.0,
            "num_tool_calls": 0.0,
            "num_graph_edits": 0.0,
            "is_submit_events": 0.0,
            "is_submit_events_batch": 0.0,
            "is_graph_edit": 0.0,
            "is_edit_op": 0.0,
            "committed": 0.0,
        }
        termination = "missing_tool_call"
        final_content = ""

        client = AsyncOpenAI(
            base_url=runtime_context.model_endpoint,
            api_key=runtime_context.api_key or "EMPTY",
            http_client=runtime_context.http_client,
            max_retries=0,
        )

        for call_idx in range(1, max_calls + 1):
            completion = await client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=self.temperature,
                stop=_TOOL_STOP_TOKENS,
                max_completion_tokens=max_tokens,
            )
            completion_ids.append(completion.id)
            content = completion.choices[0].message.content or ""
            final_content = content
            usage = TokenUsage(
                prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(completion.usage, "total_tokens", 0) or 0,
            )
            steps.append(
                TrajectoryStep(
                    step_id=f"llm-response-{call_idx}",
                    step_type=TrajectoryStepType.LLM_RESPONSE,
                    timestamp_ms=_now_ms(),
                    output={"content": content},
                    usage=usage,
                    call_id=completion.id,
                )
            )

            tool_call = _extract_tool_call(content)
            if tool_call is None:
                termination = "missing_tool_call"
                break
            stats["has_tool_call"] = 1.0
            stats["valid_json"] = 1.0

            tool_name = str(tool_call.get("name", ""))
            raw_args = tool_call.get("arguments", {})
            tool_args = raw_args if isinstance(raw_args, dict) else {}
            if tool_name == "submit_events":
                tool_name = "submit_events_batch"
                tool_args = {"events": tool_args.get("events", []), "done": True}
                stats["is_submit_events"] = 1.0
            tool_args = _normalize_tool_args(tool_name, tool_args, input_payload)
            tool_payload = {"name": tool_name, "arguments": tool_args}
            tool_payloads.append(tool_payload)
            stats["num_tool_calls"] += 1.0

            edit_op = str(tool_args.get("op", ""))
            if tool_name == "graph_edit":
                stats["is_graph_edit"] = 1.0
                stats["num_graph_edits"] += 1.0
                if edit_op in _GRAPH_EDIT_OPS:
                    stats["is_edit_op"] = 1.0
                    op_counts[edit_op] += 1
            if tool_name == "submit_events_batch":
                stats["is_submit_events_batch"] = 1.0

            steps.append(
                TrajectoryStep(
                    step_id=f"tool-call-{call_idx}",
                    step_type=TrajectoryStepType.TOOL_CALL,
                    timestamp_ms=_now_ms(),
                    input=tool_payload,
                    parent_call_id=completion.id,
                )
            )
            result = await _execute_agentm_tool(tools, tool_name, tool_args)
            if result["ok"]:
                stats["tool_ok"] = 1.0
            termination = result.get("termination") or (
                "tool_executed" if result["ok"] else "tool_error"
            )
            steps.append(
                TrajectoryStep(
                    step_id=f"tool-result-{call_idx}",
                    step_type=TrajectoryStepType.TOOL_RESULT,
                    timestamp_ms=_now_ms(),
                    output={"name": tool_name, **result},
                    parent_call_id=f"tool-call-{call_idx}",
                )
            )
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": "<tool_response>"
                    + json.dumps({"name": tool_name, **result}, ensure_ascii=False)
                    + "</tool_response>",
                }
            )
            if result.get("terminated"):
                break

        stats["committed"] = 1.0 if state.committed else 0.0
        stats["event_count"] = float(len(state.events))
        stats["edge_count"] = float(len(state.edges))
        stats["dropped_edge_count"] = float(len(state.dropped_edges))
        for op, count in op_counts.items():
            stats[f"op_{op}"] = float(count)

        status = (
            TrajectoryStatus.COMPLETED
            if stats["tool_ok"] and (state.committed or stats["num_graph_edits"] > 0)
            else TrajectoryStatus.FAILED
        )
        trajectory = Trajectory(
            trajectory_id=uuid4().hex,
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            status=status,
            final_output={
                "prediction": final_content,
                "tool_payload": tool_payloads[-1] if tool_payloads else None,
                "tool_payloads": tool_payloads,
                "termination": termination,
                "completion_ids": completion_ids,
                "events": [event.to_dict() for event in state.events],
                "edges": [edge.to_dict() for edge in state.edges],
            },
            steps=steps,
            summary_stats=stats,
            metadata={"raw_content": final_content, **agent_input.metadata},
        )
        outcome = TaskOutcome(
            sample_id=agent_input.sample_id,
            task_type=agent_input.task_type,
            prediction=trajectory.final_output,
            reference=None,
            success=status is TrajectoryStatus.COMPLETED,
            termination_reason=termination,
            metrics=stats,
            metadata=dict(agent_input.metadata),
        )
        return AgentRunResult(trajectory=trajectory, outcome=outcome)


class OpsLiteToolCallRewardStrategy(RewardStrategy):
    """Reward real AgentM extractor tool execution and graph edits."""

    async def compute(
        self,
        sample: TaskSample,
        trajectory: Trajectory,
        outcome: TaskOutcome,
        runtime_context: RuntimeContext,
    ) -> float:
        del runtime_context
        metrics = outcome.metrics or {}
        reward = 0.0
        reward += 0.10 * float(metrics.get("has_tool_call", 0.0))
        reward += 0.10 * float(metrics.get("valid_json", 0.0))
        reward += 0.20 * float(metrics.get("tool_ok", 0.0))
        reward += 0.25 * float(metrics.get("is_graph_edit", 0.0))
        reward += 0.15 * float(metrics.get("is_edit_op", 0.0))
        reward += 0.20 * float(metrics.get("committed", 0.0))
        reward += 0.10 * min(float(metrics.get("num_graph_edits", 0.0)), 3.0) / 3.0
        reward += 0.05 * float(metrics.get("is_submit_events_batch", 0.0))
        if (sample.reference or {}).get("event_count") is not None:
            reward += 0.05 * float(metrics.get("event_count_match", 0.0))
        return reward


async def _execute_agentm_tool(
    tools: Mapping[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
) -> dict[str, Any]:
    tool = tools.get(tool_name)
    if tool is None:
        return {"ok": False, "error": f"unknown AgentM extractor tool: {tool_name}"}
    try:
        outcome = await tool.execute(tool_args)
    except Exception as err:
        return {"ok": False, "error": str(err)}
    terminated = isinstance(outcome, ToolTerminate)
    result = outcome.result if terminated else outcome
    if not isinstance(result, ToolResult):
        return {"ok": False, "error": f"unexpected tool result: {type(result).__name__}"}
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    return {
        "ok": not result.is_error,
        "result": text,
        "error": text if result.is_error else None,
        "terminated": terminated,
        "termination": outcome.reason if terminated else None,
    }


def _state_from_agent_input(agent_input: AgentInput) -> ExtractionState:
    return _state_from_payload(_input_payload(agent_input))


def _state_from_payload(payload: Mapping[str, Any]) -> ExtractionState:
    recent_graph = tuple(
        Event.from_dict(_normalize_recent_event(event, payload))
        for event in payload.get("recent_graph", [])
        if isinstance(event, dict)
    )
    next_event_id = payload.get("next_event_id")
    if isinstance(next_event_id, bool) or not isinstance(next_event_id, int):
        # The ops-lite SFT rows use per-firing relative ids even when
        # recent_graph is present, so do not infer a global cursor.
        next_event_id = 1
    turn_texts = _turn_texts(payload)
    return ExtractionState(
        turn_texts=turn_texts,
        recent_graph=recent_graph,
        next_event_id=next_event_id,
    )


def _normalize_recent_event(
    event: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(event)
    normalized["external_refs"] = [
        _normalize_external_ref(ref, payload)
        for ref in event.get("external_refs") or []
        if isinstance(ref, dict)
    ]
    return normalized


def _input_payload(agent_input: AgentInput) -> dict[str, Any]:
    user = str(agent_input.context.get("user", "") or "")
    try:
        parsed = json.loads(user)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _turn_texts(payload: Mapping[str, Any]) -> dict[int, str]:
    texts: dict[int, str] = {}
    for turn in payload.get("new_turns", []) or []:
        if not isinstance(turn, dict):
            continue
        idx = turn.get("index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        texts[idx] = _render_content(turn.get("content"))
    for event in payload.get("recent_graph", []) or []:
        if not isinstance(event, dict):
            continue
        source_turns = event.get("source_turns") or []
        source_texts = event.get("source_turn_texts") or []
        if not isinstance(source_turns, list) or not isinstance(source_texts, list):
            continue
        for idx, text in zip(source_turns, source_texts, strict=False):
            if isinstance(idx, int) and isinstance(text, str):
                texts.setdefault(idx, text)
    return texts


def _render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        if isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block.get("type") == "tool_call":
            parts.append(
                f"tool_call {block.get('name')} "
                f"{json.dumps(block.get('arguments') or {}, ensure_ascii=False)}"
            )
        elif block.get("type") == "tool_result":
            parts.append(_render_content(block.get("content")))
    return "\n".join(part for part in parts if part)


def _extract_tool_call(content: str) -> dict[str, Any] | None:
    raw = content.strip()
    if "<tool_call>" in content and "</tool_call>" in content:
        raw = content.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        if not raw:
            raw = content.split("</tool_call>", 1)[1].strip()
    raw = _strip_json_fence(raw)
    parsed = _parse_jsonish(raw)
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        return None
    if "name" in parsed:
        return parsed
    if "events" in parsed:
        return {
            "name": "submit_events_batch" if "done" in parsed else "submit_events",
            "arguments": {
                "events": parsed.get("events", []),
                "done": bool(parsed.get("done", True)),
            },
        }
    return None


def _strip_json_fence(raw: str) -> str:
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return raw


def _repair_events_array(raw: str) -> str:
    """Repair a common SFT drift: {"events":[e1], e2, e3], "done":true}."""
    if '"events"' not in raw:
        return raw
    repaired = raw
    while "], {" in repaired:
        repaired = repaired.replace("], {", ", {", 1)
    return repaired


def _parse_jsonish(raw: str) -> Any | None:
    candidates = [raw]
    repaired_remove = _repair_events_array(raw)
    if repaired_remove != raw:
        candidates.append(repaired_remove)
    repaired_add = raw.replace("}], {", "}]}, {")
    if repaired_add not in candidates:
        candidates.append(repaired_add)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json5.loads(candidate)
            except Exception:
                continue
    return None


def _normalize_tool_args(
    tool_name: str,
    tool_args: dict[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if tool_name == "submit_events_batch":
        normalized = dict(tool_args)
        normalized["events"] = _normalize_events(normalized.get("events"), payload)
        return normalized
    if tool_name != "graph_edit":
        return tool_args
    normalized = dict(tool_args)
    op = str(normalized.get("op", ""))
    if op in {"add_node", "update_node"}:
        node = normalized.get("node") or normalized.get("node_update")
        if isinstance(node, dict):
            normalized["node"] = _normalize_events([node], payload)[0]
    if op in {"add_edge", "update_edge"}:
        edge = normalized.get("edge") or normalized.get("edge_update")
        if isinstance(edge, dict):
            normalized["edge"] = _normalize_edge(edge)
    return normalized


def _normalize_events(events_raw: Any, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(events_raw, list):
        return []
    events = [dict(event) for event in events_raw if isinstance(event, dict)]
    id_set = {event.get("id") for event in events if isinstance(event.get("id"), int)}
    moved_refs: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        event_id = event.get("id")
        kept_refs: list[dict[str, Any]] = []
        for ref in event.get("refs") or []:
            if not isinstance(ref, dict):
                continue
            normalized_ref = _normalize_ref(ref)
            to_id = normalized_ref.get("to")
            if isinstance(event_id, int) and isinstance(to_id, int) and to_id > event_id and to_id in id_set:
                forward_ref = dict(normalized_ref)
                forward_ref["to"] = event_id
                moved_refs.setdefault(to_id, []).append(forward_ref)
            else:
                kept_refs.append(normalized_ref)
        event["refs"] = kept_refs
        event["external_refs"] = [
            _normalize_external_ref(ref, payload)
            for ref in event.get("external_refs") or []
            if isinstance(ref, dict)
        ]
    for event in events:
        event_id = event.get("id")
        if isinstance(event_id, int) and event_id in moved_refs:
            event.setdefault("refs", []).extend(moved_refs[event_id])
    return events


def _normalize_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in ref.items()
        if key not in {"dst", "src", "src_turns", "dst_turns"}
    }
    if "to" not in normalized and isinstance(ref.get("dst"), int):
        normalized["to"] = ref["dst"]
    return normalized


def _normalize_external_ref(
    ref: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in ref.items()
        if key not in {"to_recent_graph_index", "src_turns", "dst_turns"}
    }
    if "to_recent_event_id" not in normalized:
        idx = ref.get("to_recent_graph_index")
        recent = payload.get("recent_graph") or []
        if isinstance(idx, int) and isinstance(recent, list):
            pos = idx - 1 if idx >= 1 else idx
            if 0 <= pos < len(recent) and isinstance(recent[pos], dict):
                normalized["to_recent_event_id"] = recent[pos].get("id")
    return normalized


def _normalize_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in edge.items()
        if key not in {"src_turns", "dst_turns"}
    }
    if "src" not in normalized and isinstance(edge.get("to"), int):
        normalized["src"] = edge["to"]
    if "dst" not in normalized and isinstance(edge.get("from"), int):
        normalized["dst"] = edge["from"]
    return normalized


def _target_reference(target: Any) -> dict[str, Any]:
    ref: dict[str, Any] = {}
    if not isinstance(target, Mapping):
        return ref
    messages = target.get("messages")
    if not isinstance(messages, list) or not messages:
        return ref
    msg = messages[0] or {}
    if isinstance(msg, Mapping):
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            fn = (tool_calls[0] or {}).get("function") or {}
            ref["tool_name"] = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            events = args.get("events")
            if isinstance(events, list):
                ref["event_count"] = len(events)
            return ref
    content = str((msg or {}).get("content", ""))
    payload = _extract_tool_call(content)
    if payload:
        ref["tool_name"] = payload.get("name")
        events = (payload.get("arguments") or {}).get("events")
        if isinstance(events, list):
            ref["event_count"] = len(events)
    return ref


def _now_ms() -> int:
    return int(time.time() * 1000)
