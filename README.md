# rca-autorl

Training code for the RCA agent: an AReaL workflow around DeepSeek Harness
(`dsh`), the harness bundle the agent runs in, and the reward. Design docs are
in the Notes repository under `research/ongoing/rcabench/rca/`; the harness
design is `agent/README.md`.

```text
src/autorl/     workflow, reward, fpg binding, dataset prep, train / train_sft, data/
agent/          dsh bundle (sql, take_note, submit_result) and scenario patch
configs/        fpg vocabulary, RL and SFT configs
data/sft/       distilled teacher episodes (git-lfs)
datapacks/      the corpus (not versioned)
scripts/        check.sh, run_sft.sh, run_sft_smoke.sh, run_smoke.sh, run_rl_smoke_1gpu.sh
```

## Setup

```bash
git submodule update --init --recursive        # AReaL fork Lincyaw/AReaL, branch rca-autorl
UV_HTTP_TIMEOUT=300 uv sync --python 3.12
git lfs pull                                   # data/sft/rca_sessions.jsonl
python -m autorl.harness .runs/dsh-home        # install the bundle; --reinstall after editing it
```

Behind a firewall, export `https_proxy` / `http_proxy` / `all_proxy` first and
run `git submodule sync --recursive` if the submodule url changed. Without a
proxy:

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
export npm_config_registry=https://registry.npmmirror.com
git config --global url."https://gitee.com/lincyaw/AReaL.git".insteadOf "https://github.com/Lincyaw/AReaL.git"
```

## Corpus (RL only)

Copy the files the trainers read from a machine that has `datapacks/ops-lite`,
then derive the incident prompts:

```bash
rsync -a --info=progress2 \
  --include='*/' --include='cases/*/*.parquet' \
  --include='cases/*/causal_graph_verified.json' --include='cases/*/.invalid' \
  --include='manifest.jsonl' --exclude='*' \
  <host>:<path>/datapacks/ops-lite/ datapacks/ops-lite/
python -m autorl.dataset datapacks/ops-lite       # writes train.jsonl + eval.jsonl; --check to dry-run
ls datapacks/ops-lite/cases/*/causal_graph_verified.json | wc -l   # expect 500
```

The Hub release `anon-ops/ops-lite` lacks `causal_graph_verified.json` and
cannot be used.

## SFT

RL needs a checkpoint that already submits a graph, so SFT comes first. It
needs only the checkout, the base model, and `data/sft/rca_sessions.jsonl`.

```bash
./scripts/run_sft_smoke.sh                       # one step, one GPU
./scripts/run_sft.sh                             # full epoch, 8 GPUs
./scripts/run_sft.sh train_dataset.path=... actor.path=...
```

To distill a fresh dataset from a served teacher:

```bash
export RCA_GATEWAY_BASE_URL=... RCA_GATEWAY_API_KEY=...
python -m autorl.data.collect $RCA_DATASET_ROOT/train.jsonl .runs/sft-collect/dsh-home --limit 10
python -m autorl.data.export .runs/sft-collect/dsh-home .runs/sft/rca_sessions.jsonl
```

Checkpoints land under `<fileroot>/checkpoints/<user>/<experiment>/<trial>/default/epoch*`.

## RL

```bash
export RCA_DATASET_ROOT=$PWD/datapacks/ops-lite
export AREAL_ADMIN_KEY=$(openssl rand -hex 16)
./scripts/run_smoke.sh actor.path="$CKPT" total_train_steps=200
./scripts/run_rl_smoke_1gpu.sh                   # single consumer card
```

Any AReaL config key can be overridden on the command line. What moves with
the hardware: `cluster.n_gpus_per_node`, the `d*p*t*` backend suffixes,
`train_dataset.batch_size`, `rollout.max_concurrent_rollouts`.
`gconfig.n_samples` is the group the reward weighting reads, not a throughput
knob. `econfig.difficulty=false` is the flat-score ablation arm. The config
comments explain the token-budget arithmetic.

## Validation

```bash
uv sync --dev && uv run pre-commit install
./scripts/check.sh                               # ruff, mypy, generated vocabulary.js
python -m unittest discover -s tests
```

`check.sh` and `autorl.dataset --check` pass without a GPU.
