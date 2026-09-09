"""The trainer's sampling temperature reaches the harness as the route's own."""

from __future__ import annotations

import unittest

from autorl.harness import GATEWAY_ROUTE, model_route


class RouteTest(unittest.TestCase):
    def test_the_temperature_travels_as_the_route_environment(self) -> None:
        route = model_route(
            scenario="rca", model="m", base_url="http://x", api_key="k", temperature=0.7
        )
        self.assertEqual(route.provider, GATEWAY_ROUTE)
        self.assertEqual(route.env["RCA_TEMPERATURE"], "0.7")

    def test_zero_is_a_temperature_and_none_is_none(self) -> None:
        greedy = model_route(
            scenario="rca", model="m", base_url="http://x", api_key="k", temperature=0
        )
        self.assertEqual(greedy.env["RCA_TEMPERATURE"], "0.0")
        unset = model_route(scenario="rca", model="m", base_url="http://x", api_key="k")
        self.assertNotIn("RCA_TEMPERATURE", unset.env)

    def test_the_bundle_applies_both_on_any_route(self) -> None:
        """No endpoint given, still the deepseek route, still the same variables."""
        route = model_route(
            scenario="rca", model="m", base_url="", api_key="", temperature=0.5, fork="/p"
        )
        self.assertEqual(route.env["RCA_TEMPERATURE"], "0.5")
        self.assertEqual(route.env["RCA_FORK_PREFIX"], "/p")


if __name__ == "__main__":
    unittest.main()
