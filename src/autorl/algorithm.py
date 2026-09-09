"""AReaL v2 configuration guard for the RCA training entrypoint."""

from __future__ import annotations

from autorl.interfaces import AReaLRLOOConfig


def validate_areal_v2_rloo(config: AReaLRLOOConfig) -> None:
    """Fail fast unless AReaL v2 is configured as one of the centrings of spec §3.2.

    RLOO is group leave-one-out mean with no spread; GRPO is group mean with
    group standard deviation. Anything else is a config that silently trains
    with a baseline the method does not define.
    """
    group_size = int(config.gconfig.n_samples)
    reward_norm = config.actor.reward_norm
    errors: list[str] = []
    if group_size < 2:
        errors.append("gconfig.n_samples must be >= 2")
    if config.gconfig.reward_normalization:
        errors.append("gconfig.reward_normalization must be false (actor.reward_norm decides)")
    if reward_norm is None:
        errors.append("actor.reward_norm is required")
    else:
        if reward_norm.mean_level != "group":
            errors.append("actor.reward_norm.mean_level must be group")
        rloo = reward_norm.mean_leave1out and reward_norm.std_level is None
        grpo = not reward_norm.mean_leave1out and reward_norm.std_level == "group"
        if not (rloo or grpo):
            errors.append(
                "actor.reward_norm must be RLOO (mean_leave1out, std_level null) "
                "or GRPO (group mean, std_level group)"
            )
        if int(reward_norm.group_size) != group_size:
            errors.append("actor.reward_norm.group_size must equal gconfig.n_samples")
    if config.actor.adv_norm is not None:
        errors.append("actor.adv_norm must be null")
    if float(config.actor.discount) != 1.0 or float(config.actor.gae_lambda) != 1.0:
        errors.append("actor.discount and actor.gae_lambda must both equal 1")
    if config.critic is not None:
        errors.append("critic must be null for critic-free training")
    if errors:
        raise ValueError("invalid RCA advantage config: " + "; ".join(errors))


__all__ = ["validate_areal_v2_rloo"]
