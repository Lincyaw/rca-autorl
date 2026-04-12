"""Tool environment public exports for autorl."""

__all__ = ["ToolEnvClient"]


def __getattr__(name: str):
    if name == "ToolEnvClient":
        from .client import ToolEnvClient

        return ToolEnvClient
    raise AttributeError(name)
