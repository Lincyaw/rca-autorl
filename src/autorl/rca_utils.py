from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_root_causes(value: Any) -> list[str]:
    """Extract root-cause names while preserving the first observed spelling."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in (
            "root_causes",
            "prediction",
            "nodes",
            "component",
            "service",
            "service_name",
            "component_name",
            "name",
            "container",
            "span",
            "pod",
        ):
            if key in value:
                return normalize_root_causes(value[key])
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            for root_cause in normalize_root_causes(item):
                key = root_cause.lower()
                if key not in seen:
                    normalized.append(root_cause)
                    seen.add(key)
        return normalized

    text = str(value).strip()
    return [text] if text else []


def root_cause_key_set(value: Any) -> set[str]:
    """Canonical case-insensitive key set shared by reward and outcome logic."""
    return {root_cause.lower() for root_cause in normalize_root_causes(value)}
