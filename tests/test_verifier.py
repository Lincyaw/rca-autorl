from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autorl.verifier import verify_rca


class VerifyRcaTest(unittest.TestCase):
    def test_legacy_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = verify_rca(
                {
                    "ground_truth": ["checkout-service"],
                    "fault_type": "NetworkDelay",
                },
                {
                    "root_causes": [
                        {
                            "service": "checkout-service",
                            "fault_kind": "network_delay",
                        }
                    ]
                },
                data_dir=directory,
                has_submission=True,
            )
        self.assertEqual(metrics["reward"], 1.0)
        self.assertEqual(metrics["has_submission"], 1.0)

    def test_missing_prediction_scores_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = verify_rca(
                {"ground_truth": ["mysql"], "fault_type": "cpu_stress"},
                {},
                data_dir=Path(directory),
                has_submission=False,
            )
        self.assertEqual(metrics["reward"], 0.0)
        self.assertEqual(metrics["has_submission"], 0.0)


if __name__ == "__main__":
    unittest.main()
