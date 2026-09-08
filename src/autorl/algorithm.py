"""AReaL v2 configuration guard for the RCA training entrypoint."""

from __future__ import annotations

from autorl.interfaces import AReaLRLOOConfig


def validate_areal_v2_rloo(config: AReaLRLOOConfig) -> None:
    """Fail fast unless the configuration leaves the advantage to us.

    RLOO's leave-one-out baseline is computed in `autorl.advantage`, over the
    samples of one prompt. AReaL's own normalization must therefore be off, not
    tuned: with `export_style: individual` its group is a single rollout's
    turns, so leaving it on would centre each turn against the others of its own
    episode and undo the comparison twice — once before the advantage is
    computed and once after.
    """
    group_size = int(config.gconfig.n_samples)
    errors: list[str] = []
    if group_size < 2:
        errors.append("gconfig.n_samples must be >= 2 for a leave-one-out baseline")
    if config.gconfig.reward_normalization:
        errors.append("gconfig.reward_normalization must be false (it divides by std)")
    if config.actor.reward_norm is not None:
        errors.append("actor.reward_norm must be null; autorl.advantage centres the outcome")
    if config.actor.adv_norm is not None:
        errors.append("actor.adv_norm must be null; it would re-centre what we computed")
    if float(config.actor.discount) != 1.0 or float(config.actor.gae_lambda) != 1.0:
        errors.append("actor.discount and actor.gae_lambda must both equal 1")
    if config.critic is not None:
        errors.append("critic must be null for critic-free RLOO")
    if errors:
        raise ValueError("invalid RCA RLOO config: " + "; ".join(errors))


__all__ = ["validate_areal_v2_rloo"]
