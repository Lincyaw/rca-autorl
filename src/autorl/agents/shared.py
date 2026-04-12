from __future__ import annotations

"""Compatibility shim for historical ``autorl.agents.shared`` imports.

New code should import from ``autorl.agents`` or ``autorl.agents.runner``.
"""

from .runner import AgentRunner, MultiTurnReactAgent

__all__ = ["AgentRunner", "MultiTurnReactAgent"]
