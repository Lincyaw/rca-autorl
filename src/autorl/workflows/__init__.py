"""Thin workflow exports; canonical behavior lives under autorl.runtime."""

from .agent import AReaLNativeAgentWorkflow, AgentWorkflowAdapter

__all__ = ["AReaLNativeAgentWorkflow", "AgentWorkflowAdapter"]
