"""Typed contracts at the AReaL-to-AgentM boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NotRequired, Protocol, TypedDict, Unpack, runtime_checkable

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class RCASample(TypedDict, total=False):
    id: str | int
    source: str
    incident: str
    question: str
    prompt: str
    data_dir: str
    case_dir: str
    observability_dir: str
    datapack_name: str
    data_pack_name: str
    expected_services: list[str]
    ground_truth: list[str] | dict[str, JsonValue]
    fault_kind: str
    fault_type: str


class AReaLRunOptions(TypedDict):
    base_url: str
    api_key: NotRequired[str]
    http_client: NotRequired[object]


class AgentMWorkflowConfig(TypedDict, total=False):
    scenario: str
    model: str
    max_turns: int
    timeout: float
    dataset_root: str
    reward: dict[str, float]


class RewardNormConfig(Protocol):
    mean_level: str | None
    mean_leave1out: bool
    std_level: str | None
    group_size: int


class ActorAlgorithmConfig(Protocol):
    reward_norm: RewardNormConfig | None
    adv_norm: object | None
    discount: float
    gae_lambda: float | str


class GenerationAlgorithmConfig(Protocol):
    n_samples: int
    reward_normalization: bool


class AReaLRLOOConfig(Protocol):
    actor: ActorAlgorithmConfig
    gconfig: GenerationAlgorithmConfig
    critic: object | None


@runtime_checkable
class TensorLike(Protocol):
    def detach(self) -> TensorLike: ...

    def cpu(self) -> TensorLike: ...

    def tolist(self) -> object: ...


class AReaLAgentWorkflow(ABC):
    """Typed form of AReaL v2's duck-typed agent workflow contract."""

    @abstractmethod
    async def run(
        self,
        data: RCASample,
        **extra_kwargs: Unpack[AReaLRunOptions],
    ) -> float | dict[str, float]: ...


__all__ = [
    "AReaLAgentWorkflow",
    "AReaLRLOOConfig",
    "AReaLRunOptions",
    "AgentMWorkflowConfig",
    "JsonValue",
    "RCASample",
    "TensorLike",
]
