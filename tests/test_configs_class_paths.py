"""Importability test for the llmharness training-strategy configs.

For every YAML config in ``configs/train/llmharness/`` that declares
``task_adapter_path`` / ``agent_runtime_path`` / ``reward_strategy_path``,
this test imports the referenced class and asserts it inherits from
the appropriate base. Catches dotted-path typos and class renames.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import re
import sys
import types
import unittest
from pathlib import Path


def _ensure_areal_stub() -> None:
    if "areal.utils.dynamic_import" in sys.modules:
        return
    areal_module = sys.modules.setdefault("areal", types.ModuleType("areal"))
    utils_module = sys.modules.setdefault(
        "areal.utils", types.ModuleType("areal.utils")
    )
    dynamic_import_module = types.ModuleType("areal.utils.dynamic_import")
    hf_utils_module = types.ModuleType("areal.utils.hf_utils")
    dynamic_import_module.import_from_string = lambda dotted: dotted
    hf_utils_module.load_hf_tokenizer = lambda name: name
    sys.modules["areal.utils.dynamic_import"] = dynamic_import_module
    sys.modules["areal.utils.hf_utils"] = hf_utils_module
    areal_module.utils = utils_module
    utils_module.dynamic_import = dynamic_import_module
    utils_module.hf_utils = hf_utils_module


def _ensure_datasets_stub() -> None:
    if "datasets" in sys.modules:
        return
    m = types.ModuleType("datasets")
    m.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)

    class _StubDataset:
        @classmethod
        def from_list(cls, rows):  # type: ignore[no-untyped-def]
            return rows

    m.Dataset = _StubDataset  # type: ignore[attr-defined]
    sys.modules["datasets"] = m


_ensure_areal_stub()
_ensure_datasets_stub()
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def _preload_runtime_stubs() -> None:
    """Bootstrap ``autorl.runtime`` without executing its __init__.

    The package __init__ pulls in qwen_agent / json5 / openai which
    aren't installed for unit tests. We register a bare package object
    and load only the modules we need (base + the two llmharness
    runtimes) via importlib.util.spec_from_file_location.
    """
    import importlib.util as ilu

    if "autorl.runtime" in sys.modules and hasattr(
        sys.modules["autorl.runtime"], "llmharness_extractor"
    ):
        return
    runtime_dir = src_path / "autorl" / "runtime"
    pkg = types.ModuleType("autorl.runtime")
    pkg.__path__ = [str(runtime_dir)]  # type: ignore[attr-defined]
    sys.modules["autorl.runtime"] = pkg

    for mod_name, file_name in [
        ("autorl.runtime.base", "base.py"),
        ("autorl.runtime.llmharness_extractor", "llmharness_extractor.py"),
        ("autorl.runtime.llmharness_auditor", "llmharness_auditor.py"),
    ]:
        spec = ilu.spec_from_file_location(mod_name, runtime_dir / file_name)
        assert spec and spec.loader
        m = ilu.module_from_spec(spec)
        sys.modules[mod_name] = m
        spec.loader.exec_module(m)


_preload_runtime_stubs()

# Light YAML parser using regex for the simple key: value lines we care
# about — avoids requiring PyYAML at test time.
_PATH_KEYS = ("task_adapter_path", "agent_runtime_path", "reward_strategy_path")
_PATH_LINE = re.compile(r"^(?P<key>\w+):\s*(?P<value>[\w\.]+)\s*$")


def _scan_paths(yaml_text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in yaml_text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        m = _PATH_LINE.match(line)
        if not m:
            continue
        key = m.group("key")
        if key in _PATH_KEYS:
            found[key] = m.group("value")
    return found


def _import_class(dotted: str):  # type: ignore[no-untyped-def]
    module_path, _, class_name = dotted.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


_BASES_BY_KEY = {
    "task_adapter_path": ("autorl.tasks.base", "TaskAdapter"),
    "agent_runtime_path": ("autorl.runtime.base", "AgentRuntime"),
    "reward_strategy_path": ("autorl.rewards.base", "RewardStrategy"),
}


class ConfigClassPathTests(unittest.TestCase):
    def test_all_configs_have_importable_class_paths(self) -> None:
        config_dir = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "train"
            / "llmharness"
        )
        self.assertTrue(config_dir.is_dir(), f"missing dir: {config_dir}")
        yaml_files = sorted(config_dir.glob("*.yaml"))
        self.assertGreater(len(yaml_files), 0, "no yaml configs found")

        for yaml_path in yaml_files:
            with self.subTest(config=yaml_path.name):
                paths = _scan_paths(yaml_path.read_text(encoding="utf-8"))
                for key, dotted in paths.items():
                    cls = _import_class(dotted)
                    base_mod, base_name = _BASES_BY_KEY[key]
                    base_cls = getattr(
                        importlib.import_module(base_mod), base_name
                    )
                    self.assertTrue(
                        issubclass(cls, base_cls),
                        f"{dotted} is not a subclass of {base_mod}.{base_name}",
                    )

    def test_six_configs_present(self) -> None:
        config_dir = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "train"
            / "llmharness"
        )
        names = sorted(p.name for p in config_dir.glob("*.yaml"))
        self.assertEqual(
            names,
            [
                "auditor_dpo_outcome.yaml",
                "auditor_grpo_process.yaml",
                "auditor_sft_baseline.yaml",
                "extractor_dpo_outcome.yaml",
                "extractor_grpo_process.yaml",
                "extractor_sft_baseline.yaml",
            ],
        )


if __name__ == "__main__":
    unittest.main()
