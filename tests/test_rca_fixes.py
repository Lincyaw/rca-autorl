from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
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
if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")
    openai_module.AsyncOpenAI = object
    openai_module.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
    sys.modules["openai"] = openai_module
if "transformers" not in sys.modules:
    transformers_module = types.ModuleType("transformers")
    transformers_module.PreTrainedTokenizerFast = object
    transformers_module.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    sys.modules["transformers"] = transformers_module

from autorl.contracts import RuntimeContext, Trajectory, TrajectoryStatus
from autorl.data import rcabench
from autorl.rewards.rca import RootCauseMatchRewardStrategy
from autorl.tasks.rca import RCATaskAdapter


def _load_agentm_runtime_module():
    runtime_dir = Path(__file__).resolve().parents[1] / "src" / "autorl" / "runtime"
    runtime_package = sys.modules.setdefault("autorl.runtime", types.ModuleType("autorl.runtime"))
    runtime_package.__path__ = [str(runtime_dir)]

    base_spec = importlib.util.spec_from_file_location("autorl.runtime.base", runtime_dir / "base.py")
    if base_spec is None or base_spec.loader is None:
        raise RuntimeError("failed to load autorl.runtime.base")
    base_module = importlib.util.module_from_spec(base_spec)
    sys.modules["autorl.runtime.base"] = base_module
    base_spec.loader.exec_module(base_module)

    agentm_spec = importlib.util.spec_from_file_location("autorl.runtime.agentm", runtime_dir / "agentm.py")
    if agentm_spec is None or agentm_spec.loader is None:
        raise RuntimeError("failed to load autorl.runtime.agentm")
    agentm_module = importlib.util.module_from_spec(agentm_spec)
    sys.modules["autorl.runtime.agentm"] = agentm_module
    agentm_spec.loader.exec_module(agentm_module)
    return agentm_module


agentm_runtime = _load_agentm_runtime_module()
AgentMRuntime = agentm_runtime.AgentMRuntime


class RCABenchAlignmentTests(unittest.TestCase):
    def test_load_case_bundle_uses_selected_target_graph_root_causes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir)
            (case_dir / "injection.json").write_text(
                json.dumps(
                    {
                        "injection_name": "checkout-latency",
                        "ground_truth": {"service": ["stale-service"]},
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "env.json").write_text("{}", encoding="utf-8")

            conclusion_graph = {
                "nodes": [{"component": "Fresh-Service", "state": ["anomalous"], "timestamp": ""}],
                "edges": [],
                "root_causes": [{"component": "Fresh-Service", "state": ["anomalous"], "timestamp": ""}],
                "component_to_service": {},
            }

            original_loader = rcabench._load_graph_from_conclusion
            rcabench._load_graph_from_conclusion = lambda path: conclusion_graph
            try:
                bundle = rcabench.load_case_bundle(case_dir)
            finally:
                rcabench._load_graph_from_conclusion = original_loader

        self.assertEqual(bundle["answer"], conclusion_graph)
        self.assertEqual(bundle["root_causes"], ["Fresh-Service"])


class RCATaskConsistencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_prefers_answer_and_outcome_matches_reward_case_insensitively(self) -> None:
        adapter = RCATaskAdapter()
        sample = adapter.validate_sample(
            {
                "id": "case-1",
                "incident": "Investigate the RCA case.",
                "data_dir": "/tmp/case-1",
                "root_causes": ["stale-service"],
                "answer": {
                    "root_causes": [
                        {"component": "TS-Travel-Service", "state": ["anomalous"], "timestamp": ""}
                    ]
                },
            }
        )

        trajectory = Trajectory(
            trajectory_id="traj-1",
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            status=TrajectoryStatus.COMPLETED,
            final_output={
                "prediction": {"root_causes": ["ts-travel-service"]},
                "termination": "agentm_complete",
            },
        )

        outcome = adapter.to_task_outcome(sample, trajectory)
        reward = await RootCauseMatchRewardStrategy().compute(
            sample=sample,
            trajectory=trajectory,
            outcome=outcome,
            runtime_context=RuntimeContext(mode="eval"),
        )

        self.assertEqual(sample.reference, {"root_causes": ["TS-Travel-Service"]})
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.metrics["matched_root_causes"], 1.0)
        self.assertEqual(reward, 1.0)


class AgentMRuntimeSerializationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.original_modules = {
            name: sys.modules.get(name)
            for name in ("agentm", "agentm.cli", "agentm.cli.run")
        }
        self.original_env = {
            name: os.environ.get(name)
            for name in (
                "AGENTM_API_BASE_URL",
                "AGENTM_API_KEY",
                "AGENTM_ORCHESTRATOR_MODEL",
                "AGENTM_WORKER_MODEL",
            )
        }
        self.calls: list[dict[str, str | None]] = []
        self.first_call_started = asyncio.Event()
        self.release_first_call = asyncio.Event()
        self.active_calls = 0
        self.max_active_calls = 0

        async def fake_run_investigation_headless(**_: object) -> tuple[str, str | None, str, str]:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self.calls.append(
                {
                    "base_url": os.environ.get("AGENTM_API_BASE_URL"),
                    "api_key": os.environ.get("AGENTM_API_KEY"),
                    "orchestrator_model": os.environ.get("AGENTM_ORCHESTRATOR_MODEL"),
                    "worker_model": os.environ.get("AGENTM_WORKER_MODEL"),
                }
            )
            if len(self.calls) == 1:
                self.first_call_started.set()
                await self.release_first_call.wait()
            self.active_calls -= 1
            return ('{"root_causes":["svc"]}', None, "run-id", "/tmp/traj.json")

        agentm_module = types.ModuleType("agentm")
        cli_module = types.ModuleType("agentm.cli")
        run_module = types.ModuleType("agentm.cli.run")
        run_module.run_investigation_headless = fake_run_investigation_headless
        cli_module.run = run_module
        agentm_module.cli = cli_module
        sys.modules["agentm"] = agentm_module
        sys.modules["agentm.cli"] = cli_module
        sys.modules["agentm.cli.run"] = run_module
        agentm_runtime._AGENTM_RUN_LOCK = None

    async def asyncTearDown(self) -> None:
        for name, module in self.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name, value in self.original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        agentm_runtime._AGENTM_RUN_LOCK = None

    async def test_concurrent_runs_are_serialized_while_env_is_overridden(self) -> None:
        runtime = AgentMRuntime()

        task_one = asyncio.create_task(
            runtime.run(
                agent_input=adapter_input("sample-1", "/tmp/case-1"),
                runtime_context=RuntimeContext(
                    mode="eval",
                    model_endpoint="http://endpoint-one",
                    api_key="key-one",
                    metadata={
                        "agentm_orchestrator_model": "orchestrator-one",
                        "agentm_worker_model": "worker-one",
                    },
                ),
            )
        )
        await self.first_call_started.wait()

        task_two = asyncio.create_task(
            runtime.run(
                agent_input=adapter_input("sample-2", "/tmp/case-2"),
                runtime_context=RuntimeContext(
                    mode="eval",
                    model_endpoint="http://endpoint-two",
                    api_key="key-two",
                    metadata={
                        "agentm_orchestrator_model": "orchestrator-two",
                        "agentm_worker_model": "worker-two",
                    },
                ),
            )
        )
        await asyncio.sleep(0.05)

        self.assertEqual(self.max_active_calls, 1)
        self.assertEqual(len(self.calls), 1)

        self.release_first_call.set()
        await asyncio.gather(task_one, task_two)

        self.assertEqual(
            [call["base_url"] for call in self.calls],
            ["http://endpoint-one", "http://endpoint-two"],
        )
        self.assertEqual(
            [call["api_key"] for call in self.calls],
            ["key-one", "key-two"],
        )
        self.assertEqual(
            [call["orchestrator_model"] for call in self.calls],
            ["orchestrator-one", "orchestrator-two"],
        )
        self.assertEqual(
            [call["worker_model"] for call in self.calls],
            ["worker-one", "worker-two"],
        )
        for name, value in self.original_env.items():
            self.assertEqual(os.environ.get(name), value)


def adapter_input(sample_id: str, data_dir: str):
    from autorl.contracts import AgentInput

    return AgentInput(
        sample_id=sample_id,
        task_type="rca",
        context={"incident": f"Investigate {sample_id}", "data_dir": data_dir},
        metadata={},
    )


if __name__ == "__main__":
    unittest.main()
