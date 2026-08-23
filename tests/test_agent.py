from __future__ import annotations

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from autorl.agent import AgentMWorkflow, resolve_data_dir


class ResolveDataDirTest(unittest.TestCase):
    def test_uses_explicit_data_dir(self) -> None:
        self.assertEqual(resolve_data_dir({"data_dir": "/cases/one"}), "/cases/one")

    def test_joins_datapack_to_dataset_root(self) -> None:
        self.assertEqual(
            resolve_data_dir({"datapack_name": "case-1"}, "/dataset"),
            "/dataset/case-1",
        )

    def test_requires_case_location(self) -> None:
        with self.assertRaises(ValueError):
            resolve_data_dir({"question": "what failed?"})


class AgentMWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def test_passes_areal_proxy_directly_to_agentm(self) -> None:
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def run(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    response=('{"root_causes":[{"service":"mysql","fault_kind":"cpu_stress"}]}'),
                    metadata={"submit_final_report_seen": True},
                )

        package = ModuleType("rca_eval")
        module = ModuleType("rca_eval.agent")
        module.AgentMAgent = FakeAgent
        with patch.dict("sys.modules", {"rca_eval": package, "rca_eval.agent": module}):
            reward = await AgentMWorkflow({"model": "policy"}).run(
                {
                    "question": "what failed?",
                    "data_dir": "/missing-is-ok-for-legacy-score",
                    "ground_truth": ["mysql"],
                    "fault_type": "cpu_stress",
                },
                base_url="http://rollout",
                api_key="secret",
            )

        provider_name, provider_config = captured["provider_tuple"]
        self.assertEqual(provider_name, "agentm.extensions.builtin.llm_openai")
        self.assertEqual(provider_config["base_url"], "http://rollout")
        self.assertEqual(provider_config["api_key"], "secret")
        self.assertEqual(reward["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
