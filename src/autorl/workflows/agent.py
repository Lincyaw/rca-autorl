from __future__ import annotations

from autorl.runtime.agent_workflow import UnifiedAgentWorkflow


class AReaLNativeAgentWorkflow(UnifiedAgentWorkflow):
    """Canonical workflow alias for repo-level imports and config defaults."""


# Canonical project alias.
AgentWorkflowAdapter = AReaLNativeAgentWorkflow
