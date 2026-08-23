"""SFT loss-mask correctness against a real Qwen3 Thinking tokenizer.

Under the v19 extractor flow, an SFT row's ``target.messages`` is a
multi-turn ``[assistant, tool, assistant, ...]`` sequence. Tool messages
are deterministic witness output that the student must NOT learn to
generate — supervising tool tokens would be wrong signal.

This test pins the expected behaviour of
:func:`autorl.data.sft._convert_sample`:

* tokens at ``loss_mask == 1`` positions decode to assistant-role content
  only (``<think>`` blocks + tool_call payload),
* tokens at ``loss_mask == 0`` positions decode to system + user prompt
  + tool-role spans (the ``<tool_response>`` wrapper and inner body), and
  the assistant-role headers within the first asst turn that the
  template would have produced under ``add_generation_prompt=True``.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
import unittest
from pathlib import Path


def _ensure_stubs() -> None:
    if "datasets" not in sys.modules:
        datasets_module = types.ModuleType("datasets")
        datasets_module.__spec__ = importlib.machinery.ModuleSpec(
            "datasets", loader=None
        )

        class _StubDataset:
            @classmethod
            def from_list(cls, rows):  # type: ignore[no-untyped-def]
                return rows

        datasets_module.Dataset = _StubDataset  # type: ignore[attr-defined]
        sys.modules["datasets"] = datasets_module


_ensure_stubs()

src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from autorl.data.sft import _convert_sample

_QWEN3_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


def _load_qwen3_tokenizer():  # type: ignore[no-untyped-def]
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise unittest.SkipTest(f"transformers not installed: {exc}") from None
    try:
        return AutoTokenizer.from_pretrained(_QWEN3_MODEL_ID, local_files_only=True)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - optional cache
        raise unittest.SkipTest(
            f"cannot load {_QWEN3_MODEL_ID}: {exc}"
        ) from None


class Qwen3LossMaskTests(unittest.TestCase):
    def test_only_assistant_spans_supervised_in_multi_turn_target(self) -> None:
        tokenizer = _load_qwen3_tokenizer()
        sample = {
            "phase": "extractor",
            "sample_id": "case-x",
            "root_session_id": "rsid",
            "turn_index": 0,
            "input": {
                "system": "you are an extractor",
                "user": '{"firing": "x"}',
            },
            "target": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "<think>plan first edit</think>\n\n",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "upsert_node",
                                    "arguments": '{"id":"n1"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "content": "ok node n1"},
                    {
                        "role": "assistant",
                        "content": "<think>now finalize</think>\n\n",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "finalize_extraction",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                ]
            },
            "meta": {},
        }

        out = _convert_sample(sample, tokenizer=tokenizer)
        ids = out["input_ids"]
        mask = out["loss_mask"]
        self.assertEqual(len(ids), len(mask))
        self.assertTrue(any(m == 1 for m in mask), "no tokens supervised")
        self.assertTrue(any(m == 0 for m in mask), "no tokens unsupervised")

        supervised_ids = [tid for tid, m in zip(ids, mask) if m == 1]
        unsupervised_ids = [tid for tid, m in zip(ids, mask) if m == 0]
        supervised = tokenizer.decode(supervised_ids)
        unsupervised = tokenizer.decode(unsupervised_ids)

        # Assistant-role evidence must land in mask=1.
        self.assertIn("plan first edit", supervised)
        self.assertIn("now finalize", supervised)
        self.assertIn("upsert_node", supervised)
        self.assertIn("finalize_extraction", supervised)

        # The system + user prompt body must be in mask=0.
        self.assertIn("you are an extractor", unsupervised)
        self.assertIn('{"firing": "x"}', unsupervised)

        # The tool-role wrapper + body must be in mask=0. Qwen renders
        # tool messages as ``<|im_start|>user\n<tool_response>...``, so
        # the literal ``<tool_response>`` marker is the strongest signal
        # that the tool span was correctly excluded.
        self.assertIn("<tool_response>", unsupervised)
        self.assertIn("ok node n1", unsupervised)
        self.assertNotIn("<tool_response>", supervised)
        self.assertNotIn("ok node n1", supervised)

        # System / user content must NOT leak into the supervised range.
        self.assertNotIn("you are an extractor", supervised)
        self.assertNotIn('{"firing": "x"}', supervised)


if __name__ == "__main__":
    unittest.main()
