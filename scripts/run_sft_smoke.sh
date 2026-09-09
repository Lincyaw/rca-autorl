#!/usr/bin/env bash
# One SFT step on one GPU: the full config with the run cut short. Any extra
# args are forwarded, e.g. `./scripts/run_sft_smoke.sh actor.path=Qwen/Qwen3-8B`.
set -euo pipefail

exec "$(dirname -- "${BASH_SOURCE[0]}")/run_sft.sh" \
  experiment_name=autorl-dsh-rca-sft-smoke trial_name=local-smoke \
  total_train_steps=1 cluster.n_gpus_per_node=1 actor.backend=fsdp:d1p1t1 \
  cluster.fileroot=.runs/dsh-rca-sft-smoke \
  cluster.name_resolve.nfs_record_root=.runs/dsh-rca-sft-smoke/name_resolve \
  train_dataset.batch_size=1 valid_dataset.batch_size=1 \
  "$@"
