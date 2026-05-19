"""No-GPU smoke for the SFT data pipeline.

Loads ``autorl.data.sft.build_sft_dataset_from_manifest`` on a real
llmharness/distill ``extractor.jsonl`` with a Qwen3 tokenizer, picks the
four shortest rows, then runs a handful of optimizer steps with
Qwen3-0.6B on CPU and prints per-step loss / grad-norm.

What this proves:

* Chat template + ``loss_mask`` actually produce a usable LM signal —
  loss on a previously-seen row drops on the next pass (within
  ``n_steps`` we cycle 4 rows × ~1.5 epochs).
* No OOM, NaN, or template-misalignment surprises before burning GPU
  time on the real AReaL ``SFTTrainer`` path.

What this does NOT prove:

* The AReaL FSDP wiring in ``agent_sft/train.py`` (needs CUDA — run
  ``scripts/run_sft_smoke.sh`` for that).
* That the chosen Qwen3-4B-Thinking checkpoint converges; we verify
  with a smaller sibling that shares the same tokenizer / template.

Defaults (overridable via env vars):

* ``SFT_SMOKE_MANIFEST`` — path to the distill ``extractor.jsonl``
* ``SFT_SMOKE_MODEL`` — HF model id, default ``Qwen/Qwen3-0.6B``
* ``SFT_SMOKE_STEPS`` — number of optimizer steps, default ``6``

Wall-clock on an 8-core CPU: ~7 min total (1 model load + 6 × ~60s).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autorl.data.sft import build_sft_dataset_from_manifest  # noqa: E402

DEFAULT_MANIFEST = (
    ROOT
    / ".."
    / "AgentM"
    / "contrib"
    / "extensions"
    / "llmharness"
    / "runs"
    / "sft-10case-2026-05-18"
    / "extractor.jsonl"
)


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    manifest = Path(os.environ.get("SFT_SMOKE_MANIFEST", str(DEFAULT_MANIFEST)))
    model_id = os.environ.get("SFT_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
    n_steps = int(os.environ.get("SFT_SMOKE_STEPS", "6"))
    if not manifest.exists():
        raise FileNotFoundError(
            f"distill manifest not found: {manifest}\n"
            "Override with SFT_SMOKE_MANIFEST=/path/to/extractor.jsonl"
        )

    log(f"manifest: {manifest}")
    log(f"model:    {model_id}")
    log(f"steps:    {n_steps}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = build_sft_dataset_from_manifest(
        str(manifest),
        tokenizer,
        max_length=32768,
    )

    shortest = sorted(range(len(ds)), key=lambda i: len(ds[i]["input_ids"]))[:4]
    log(f"using rows {shortest} from dataset of {len(ds)}")
    for i in shortest:
        row = ds[i]
        log(
            f"  row {i}: tokens={len(row['input_ids'])}  "
            f"supervised={sum(row['loss_mask'])}"
        )

    device = torch.device("cpu")
    dtype = torch.float32
    log(f"loading {model_id} on {device} ({dtype}) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    log(
        f"  model loaded in {time.time()-t0:.1f}s "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses: list[float] = []
    for step in range(n_steps):
        idx = shortest[step % len(shortest)]
        row = ds[idx]
        ids = torch.tensor([row["input_ids"]], device=device)
        mask = torch.tensor([row["loss_mask"]], device=device, dtype=torch.float32)

        t0 = time.time()
        logits = model(ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        denom = shift_mask.sum().clamp(min=1.0)
        loss = (token_loss * shift_mask).sum() / denom

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        elapsed = time.time() - t0
        loss_val = float(loss.detach())
        losses.append(loss_val)
        log(
            f"  step {step}: row={idx:>3} toks={ids.shape[1]:>5} "
            f"sup={int(denom.item()):>4} loss={loss_val:.4f} "
            f"grad={float(grad_norm):.3f}  {elapsed:.1f}s"
        )

    log("")
    log(f"loss trajectory: {[round(x, 4) for x in losses]}")
    log(
        f"first→last: {losses[0]:.4f} → {losses[-1]:.4f}  "
        f"(Δ={losses[-1] - losses[0]:+.4f})"
    )
    if len(losses) >= 2 * len(shortest):
        head = len(shortest)
        log("same-row comparison (first pass vs second pass):")
        for offset, idx in enumerate(shortest):
            before = losses[offset]
            after = losses[offset + head]
            log(f"  row {idx}: {before:.4f} -> {after:.4f}  (Δ={after - before:+.4f})")


if __name__ == "__main__":
    main()
