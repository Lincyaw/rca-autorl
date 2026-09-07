#!/usr/bin/env bash
# One RL step end to end on a single consumer GPU. Verified on an RTX 5090
# (SM120) with Qwen3-0.6B: rollout through the harness, logp, PPO update, and
# the weight push back to sglang. Reward is still the placeholder 0.0, so the
# advantage is exactly zero and nothing is learned -- this checks plumbing,
# not learning.
#
# Every override below exists because the repo default does not hold on one
# consumer GPU; on a multi-GPU datacenter node run scripts/run_smoke.sh instead.
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

DSH_HOME=${DSH_HOME:-$ROOT_DIR/.runs/dsh-live-smoke/dsh-home}
python3 -m autorl.harness "$DSH_HOME"

# sglang JIT-compiles kernels with nvcc, and CUDA < 12.8 cannot target SM120.
# Point CUDA_HOME at a toolkit new enough for the GPU when the system one is older.
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi
# Keep the CUDA runtime that torch was built against ahead of any system copy.
export LD_LIBRARY_PATH="$(ls -d "$ROOT_DIR"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
# sglang and FSDP share one device; without this they fragment each other out of memory.
export PYTORCH_ALLOC_CONF=expandable_segments:True

python3 -m autorl.train --config configs/train/dsh_rca_smoke.yaml \
  total_train_steps=1 \
  cluster.fileroot=.runs/dsh-live-smoke \
  cluster.name_resolve.nfs_record_root=.runs/dsh-live-smoke/name_resolve \
  econfig.dsh_home="$DSH_HOME" \
  actor.path=Qwen/Qwen3-0.6B \
  gconfig.n_samples=2 actor.reward_norm.group_size=2 \
  gconfig.max_new_tokens=1024 \
  sglang.context_length=8192 \
  `# A microbatch must hold the longest packed sequence, and the logits for one` \
  `# are vocab-sized, so this is the largest value that still fits beside sglang.` \
  actor.mb_spec.max_tokens_per_mb=8192 ref.mb_spec.max_tokens_per_mb=8192 \
  `# Leave the rest of the device to FSDP; sglang never gives KV cache back.` \
  sglang.mem_fraction_static=0.15 \
  `# fa3 needs SM<=90; flashinfer's JIT needs nvcc >= 12.9 to target SM120.` \
  +sglang.attention_backend=triton +sglang.sampling_backend=pytorch \
  `# The xccl weight sync builds a 2-rank NCCL group across actor and sglang;` \
  `# NCCL rejects two ranks on one device, so hand the weights over via disk.` \
  +actor.weight_update_mode=disk \
  `# The gateway binds a routable address and refuses the documented default key.` \
  +rollout.agent.admin_api_key="${AREAL_ADMIN_KEY:-rca-local-smoke-key}" \
  "$@"
