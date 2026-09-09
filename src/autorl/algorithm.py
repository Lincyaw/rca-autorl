"""AReaL v2 configuration guard for the RCA training entrypoint."""

from __future__ import annotations

from autorl.interfaces import AReaLRLOOConfig


def validate_areal_v2_rloo(config: AReaLRLOOConfig) -> None:
    """Fail fast unless AReaL v2 leaves the advantage to the workflow.

    The centring of spec §3.2 (RLOO, GRPO, ReMax) is computed per trajectory in
    `DshWorkflow.rescore_group`, together with the fork advantages of §4, so
    the framework must add no baseline of its own: a group normalization here
    would centre row against row, weighted by trajectory length, on top of it.
    """
    group_size = int(config.gconfig.n_samples)
    errors: list[str] = []
    if group_size < 2:
        errors.append("gconfig.n_samples must be >= 2")
    if config.gconfig.reward_normalization:
        errors.append("gconfig.reward_normalization must be false")
    if config.actor.reward_norm is not None:
        errors.append("actor.reward_norm must be null: the workflow centres the group itself")
    if config.actor.adv_norm is not None:
        errors.append("actor.adv_norm must be null")
    if config.econfig.centring not in {"rloo", "grpo", "remax"}:
        errors.append("econfig.centring must be rloo, grpo, or remax")
    if float(config.actor.discount) != 1.0 or float(config.actor.gae_lambda) != 1.0:
        errors.append("actor.discount and actor.gae_lambda must both equal 1")
    if config.critic is not None:
        errors.append("critic must be null for critic-free training")
    if errors:
        raise ValueError("invalid RCA advantage config: " + "; ".join(errors))


__all__ = ["validate_areal_v2_rloo"]
