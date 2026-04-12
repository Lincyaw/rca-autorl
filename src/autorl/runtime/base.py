from __future__ import annotations

from abc import ABC, abstractmethod

from autorl.contracts import AgentInput, AgentRunResult, RuntimeContext


class AgentRuntime(ABC):
    """Framework-agnostic runtime interface for one agent episode."""

    @abstractmethod
    async def run(
        self,
        agent_input: AgentInput,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        """Execute one episode and return canonical trajectory + outcome."""
        raise NotImplementedError
