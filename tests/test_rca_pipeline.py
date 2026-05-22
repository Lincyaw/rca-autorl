"""Minimal coverage for the new RCA pipeline.

Covers the wire-level contracts that the rest of the stack relies on:

* :class:`autorl.tasks.rca.RCATaskAdapter` accepts the new
  ``expected_services + fault_kind`` sample shape, exposes them as
  ``TaskSample.reference``, and emits ``service_hit`` /
  ``fault_kind_hit`` / ``score`` on ``TaskOutcome.metrics`` after
  reading the AgentRCAOutput-shaped ``prediction`` from the
  trajectory's ``final_output``.

* :class:`autorl.rewards.rca.RCABaselineRewardStrategy` returns the
  composite ``reward`` plus the dimension dict AReaL's
  ``stats_tracker`` consumes.

* :func:`autorl.data.sft._convert_sample` turns one llmharness/distill
  row into ``input_ids + loss_mask`` and the ``loss_mask`` only fires
  over the assistant turn (including the ``<think>`` block and the
  tool_call JSON).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


def _ensure_areal_stub() -> None:
    if "areal.utils.dynamic_import" in sys.modules:
        return
    areal_module = sys.modules.setdefault("areal", types.ModuleType("areal"))
    utils_module = sys.modules.setdefault("areal.utils", types.ModuleType("areal.utils"))
    dynamic_import_module = types.ModuleType("areal.utils.dynamic_import")
    hf_utils_module = types.ModuleType("areal.utils.hf_utils")
    dynamic_import_module.import_from_string = lambda dotted_path: dotted_path
    hf_utils_module.load_hf_tokenizer = lambda name: name
    sys.modules["areal.utils.dynamic_import"] = dynamic_import_module
    sys.modules["areal.utils.hf_utils"] = hf_utils_module
    areal_module.utils = utils_module
    utils_module.dynamic_import = dynamic_import_module
    utils_module.hf_utils = hf_utils_module


_ensure_areal_stub()
if "datasets" not in sys.modules:
    datasets_module = types.ModuleType("datasets")
    datasets_module.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)

    class _StubDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    datasets_module.Dataset = _StubDataset
    sys.modules["datasets"] = datasets_module

src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from autorl.contracts import RuntimeContext, Trajectory, TrajectoryStatus  # noqa: E402
from autorl.rewards.rca import RCABaselineRewardStrategy  # noqa: E402
from autorl.tasks.rca import RCATaskAdapter  # noqa: E402


class _StubTokenizer:
    """Mimics HF tokenizer.apply_chat_template enough for unit-level checks.

    Prefix-stable across turns so the multi-turn loss-mask builder can
    derive per-message spans (real Qwen / GLM thinking templates have
    this property by construction).
    """

    _GEN_HEADER = (253,)  # mimics ``<|im_start|>assistant\n<think>\n``
    _ROLE_SEP = 254

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False):
        ids: list[int] = []
        for idx, m in enumerate(messages):
            role = (m.get("role") or "").lower()
            # The first assistant message's "header" (gen prompt) collapses
            # into the prompt baseline iff a previous tokenization passed
            # add_generation_prompt=True — see Qwen behaviour. We emit it
            # only when transitioning from a non-assistant context to an
            # assistant message, so subsequent tokenizations that include
            # this message reproduce the same prefix.
            if role == "assistant":
                prev_role = (messages[idx - 1].get("role") if idx > 0 else "") or ""
                if prev_role.lower() != "assistant":
                    ids.extend(self._GEN_HEADER)
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls") or []
            for ch in content:
                ids.append(ord(ch) % 256)
            for tc in tool_calls:
                arg_text = (tc.get("function") or {}).get("arguments") or ""
                for ch in str(arg_text):
                    ids.append(ord(ch) % 256)
            ids.append(self._ROLE_SEP)
        if add_generation_prompt:
            # Last message in the prompt is always non-assistant (system/user/tool);
            # emit the same header that the first asst chunk would otherwise emit,
            # so prompt+gen is a strict prefix of prompt+1asst (no gen).
            ids.extend(self._GEN_HEADER)
        return ids


class RCATaskAdapterTests(unittest.TestCase):
    def test_validate_sample_accepts_explicit_shape(self) -> None:
        adapter = RCATaskAdapter()
        sample = adapter.validate_sample(
            {
                "id": "ts9-case",
                "incident": "checkout latency spikes",
                "data_dir": "/tmp/case",
                "expected_services": ["ts-order-service", "ts-cart-service"],
                "fault_kind": "cpu_stress",
            }
        )
        self.assertEqual(sample.sample_id, "ts9-case")
        self.assertEqual(
            sample.reference["expected_services"],
            ["ts-order-service", "ts-cart-service"],
        )
        self.assertEqual(sample.reference["fault_kind"], "cpu_stress")

    def test_validate_sample_accepts_agentm_dataset_shape(self) -> None:
        adapter = RCATaskAdapter()
        os.environ["AGENTM_RCA_DATASET_ROOT"] = "/home/ddq/AoyangSpace/dataset/rca"
        try:
            sample = adapter.validate_sample(
                {
                    "id": 5,
                    "source": "ts0-mysql-corrupt-kwx8n5",
                    "question": "API endpoints have SLO violations; investigate.",
                    "answer": "mysql,ts-station-service",
                    "ground_truth": ["mysql", "ts-station-service"],
                    "fault_type": "NetworkCorrupt",
                    "datapack_name": "ts0-mysql-corrupt-kwx8n5",
                    "tags": ["fse"],
                }
            )
        finally:
            os.environ.pop("AGENTM_RCA_DATASET_ROOT", None)
        self.assertEqual(sample.sample_id, "ts0-mysql-corrupt-kwx8n5")
        self.assertEqual(
            sample.input["data_dir"],
            "/home/ddq/AoyangSpace/dataset/rca/ts0-mysql-corrupt-kwx8n5",
        )
        self.assertEqual(
            sample.reference["expected_services"],
            ["mysql", "ts-station-service"],
        )
        self.assertEqual(sample.reference["fault_kind"], "NetworkCorrupt")

    def test_validate_sample_requires_dataset_root_for_datapack(self) -> None:
        adapter = RCATaskAdapter()
        os.environ.pop("AGENTM_RCA_DATASET_ROOT", None)
        with self.assertRaises(ValueError):
            adapter.validate_sample(
                {
                    "source": "case",
                    "question": "x",
                    "datapack_name": "case",
                }
            )

    def test_to_task_outcome_scores_against_reference(self) -> None:
        adapter = RCATaskAdapter()
        sample = adapter.validate_sample(
            {
                "id": "ts9-case",
                "incident": "x",
                "data_dir": "/tmp/case",
                "expected_services": ["ts-order-service"],
                "fault_kind": "cpu_stress",
            }
        )
        trajectory = Trajectory(
            trajectory_id="tid",
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            status=TrajectoryStatus.COMPLETED,
            final_output={
                "prediction": {
                    "root_causes": [
                        {
                            "service": "ts-order-service",
                            "fault_kind": "CPU stress (high load)",
                        }
                    ],
                    "propagation": [],
                },
                "termination": "agentm_complete",
            },
            summary_stats={"has_submission": 1.0},
        )
        outcome = adapter.to_task_outcome(sample, trajectory)
        self.assertEqual(outcome.metrics["service_hit"], 1.0)
        self.assertEqual(outcome.metrics["fault_kind_hit"], 1.0)
        self.assertAlmostEqual(outcome.metrics["score"], 1.0)
        self.assertTrue(outcome.success)


class RCABaselineRewardTests(unittest.IsolatedAsyncioTestCase):
    async def test_reward_returns_score_and_dimensions(self) -> None:
        adapter = RCATaskAdapter()
        sample = adapter.validate_sample(
            {
                "id": "case",
                "incident": "x",
                "data_dir": "/tmp/case",
                "expected_services": ["svc-A"],
                "fault_kind": "net_delay",
            }
        )
        trajectory = Trajectory(
            trajectory_id="tid",
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            status=TrajectoryStatus.COMPLETED,
            final_output={
                "prediction": {
                    "root_causes": [{"service": "svc-A", "fault_kind": "OTHER"}],
                },
                "termination": "agentm_complete",
            },
            summary_stats={"has_submission": 1.0},
        )
        outcome = adapter.to_task_outcome(sample, trajectory)
        reward = await RCABaselineRewardStrategy().compute(
            sample=sample,
            trajectory=trajectory,
            outcome=outcome,
            runtime_context=RuntimeContext(mode="eval"),
        )
        self.assertAlmostEqual(reward["reward"], 0.7)
        self.assertEqual(reward["service_hit"], 1.0)
        self.assertEqual(reward["fault_kind_hit"], 0.0)


class SftLoaderTests(unittest.TestCase):
    def test_convert_sample_emits_loss_mask_over_assistant(self) -> None:
        from autorl.data.sft import _convert_sample

        row = {
            "phase": "extractor",
            "sample_id": "case",
            "root_session_id": "rsid",
            "turn_index": 0,
            "input": {
                "system": "you are an extractor",
                "user": '{"new_turns": [{"index": 0}]}',
            },
            "target": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "<think>thinking</think>\n\n",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "submit_events",
                                    "arguments": '{"events":[]}',
                                },
                            }
                        ],
                    }
                ]
            },
            "meta": {},
        }
        out = _convert_sample(row, tokenizer=_StubTokenizer())
        self.assertEqual(len(out["input_ids"]), len(out["loss_mask"]))
        self.assertGreater(sum(out["loss_mask"]), 0)
        self.assertEqual(out["loss_mask"][0], 0)
        self.assertEqual(out["loss_mask"][-1], 1)


if __name__ == "__main__":
    unittest.main()
