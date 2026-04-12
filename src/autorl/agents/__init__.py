"""Canonical agent exports for autorl.

Use ``autorl.agents.AgentRunner`` as the stable package-level import path.
``autorl.agents.shared`` remains a compatibility shim for historical callers.
"""

__all__ = ["AgentRunner", "MultiTurnReactAgent"]


def __getattr__(name: str):
    if name == "AgentRunner":
        from .runner import AgentRunner

        return AgentRunner
    if name == "MultiTurnReactAgent":
        from .runner import MultiTurnReactAgent

        return MultiTurnReactAgent
    raise AttributeError(name)
