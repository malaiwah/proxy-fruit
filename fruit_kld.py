#!/usr/bin/env python3
"""Full-vocabulary forward-KL baseline: Fruit trainer graph versus vLLM.

Run in two fresh processes because the BF16 trainer and served SIQ model each
need the GPU:

  PHASE=ref CKPT=... REF_PT=... MOE_IMPL=grouped ROPE_THETA=500000 \
    python fruit_kld.py
  PHASE=serve SERVED=... REF_PT=... python fruit_kld.py

The candidate phase requests every vocabulary log-probability from vLLM. It
computes exact categorical KL(P_ref || P_served), not a top-K proxy.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from checkpoint_contract import checkpoint_model_state

DEFAULT_PROMPT = "Once upon a time, there was"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _small_file_hashes(root: Path) -> dict[str, str]:
    names = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "MANIFEST.sha256",
    )
    return {name: _sha256(root / name) for name in names if (root / name).is_file()}


def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=False)


def _reference_phase(prompt: str, ref_path: Path) -> None:
    import train_fruit as tf
    from parity_test import _reference_conventions

    workspace_config = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)

    checkpoint = Path(os.environ["CKPT"])
    tokenizer_path = os.environ.get("TOK_DIR", "/mnt/vault/llm/glm52-franken/src")
    tokenizer = _tokenizer(tokenizer_path)
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) < 2:
        raise ValueError("KLD prompt must contain at least two tokens")

    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint_model_state(payload)
    native, theta = _reference_conventions(state, os.environ)
    tf.CONV["serve"] = native
    tf.THETA = theta
    model = tf.Fruit()
    model.load_state_dict(state, strict=True)
    del payload, state
    model = model.to("cuda:0", torch.bfloat16).eval()
    inputs = torch.tensor([token_ids], device="cuda:0")
    with (
        torch.inference_mode(),
        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
    ):
        hidden, _ = model(inputs)
        logits = model.lm_head(hidden[0, :-1]).float()
        reference_log_probs = torch.log_softmax(logits, dim=-1).cpu()
    del model, hidden, logits, inputs
    torch.cuda.empty_cache()

    reference = {
        "kind": "fruit_full_vocabulary_kld_reference",
        "schema_version": SCHEMA_VERSION,
        "prompt": prompt,
        "token_ids": token_ids,
        "vocab_size": int(reference_log_probs.shape[1]),
        "positions": int(reference_log_probs.shape[0]),
        "log_probs": reference_log_probs,
        "metadata": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "tokenizer": str(Path(tokenizer_path).resolve()),
            "tokenizer_hashes": _small_file_hashes(Path(tokenizer_path)),
            "serving_conventions": native,
            "rope_theta": theta,
            "moe_impl": os.environ.get("MOE_IMPL", ""),
            "deterministic_algorithms": True,
            "sdpa_backend": "math",
            "cublas_workspace_config": workspace_config,
            "geometry": {
                "hidden": tf.H,
                "layers": tf.NL,
                "heads": tf.HEADS,
                "q_lora": tf.Q_LORA,
                "dense_intermediate": tf.DENSE_INTER,
                "moe_intermediate": tf.MOE_INTER,
            },
            "torch_version": torch.__version__,
        },
    }
    torch.save(reference, ref_path)
    print(
        "FRUIT-KLD-REF "
        + json.dumps(
            {key: value for key, value in reference.items() if key != "log_probs"},
            sort_keys=True,
        ),
        flush=True,
    )


def _dense_served_log_probs(values: dict[int, Any], vocab_size: int) -> torch.Tensor:
    if len(values) != vocab_size:
        raise ValueError(
            f"served distribution has {len(values)} entries, "
            f"expected full vocabulary {vocab_size}"
        )
    token_ids = torch.tensor(tuple(values), dtype=torch.long)
    if (
        token_ids.numel() != vocab_size
        or int(token_ids.min()) != 0
        or int(token_ids.max()) != vocab_size - 1
        or torch.unique(token_ids).numel() != vocab_size
    ):
        raise ValueError("served distribution token IDs do not close 0..vocab-1")
    log_probs = torch.empty(vocab_size, dtype=torch.float64)
    log_probs[token_ids] = torch.tensor(
        [values[int(token_id)].logprob for token_id in token_ids],
        dtype=torch.float64,
    )
    return log_probs - torch.logsumexp(log_probs, dim=0)


def _serve_phase(ref_path: Path) -> None:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    reference = torch.load(ref_path, map_location="cpu", weights_only=False)
    if (
        reference.get("kind") != "fruit_full_vocabulary_kld_reference"
        or reference.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported Fruit KLD reference schema")
    ref_log_probs = reference["log_probs"]
    token_ids = list(reference["token_ids"])
    vocab_size = int(reference["vocab_size"])
    positions = int(reference["positions"])
    if tuple(ref_log_probs.shape) != (positions, vocab_size):
        raise ValueError("reference log-probability shape does not match metadata")
    if positions != len(token_ids) - 1:
        raise ValueError("reference positions do not align with prompt tokens")

    served = Path(os.environ["SERVED"])
    llm = LLM(
        model=str(served),
        tensor_parallel_size=1,
        max_model_len=max(256, len(token_ids) + 1),
        max_num_batched_tokens=max(256, len(token_ids) + 1),
        max_num_seqs=1,
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.45")),
        kv_cache_dtype=os.environ.get("KV", "fp8_ds_mla"),
        enforce_eager=True,
        trust_remote_code=False,
        max_logprobs=-1,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=-1)
    output = llm.generate(
        [TokensPrompt(prompt_token_ids=token_ids)], sampling, use_tqdm=False
    )[0]
    if list(output.prompt_token_ids) != token_ids:
        raise ValueError("served prompt token IDs differ from the reference")
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(token_ids):
        raise ValueError("served prompt log-probabilities are incomplete")

    per_position: list[dict[str, float | int]] = []
    for served_position in range(1, len(token_ids)):
        values = prompt_logprobs[served_position]
        if values is None:
            raise ValueError(
                f"missing served log-probabilities at position {served_position}"
            )
        candidate = _dense_served_log_probs(values, vocab_size)
        teacher = ref_log_probs[served_position - 1].to(torch.float64)
        teacher = teacher - torch.logsumexp(teacher, dim=0)
        probability = teacher.exp()
        forward_kl = float((probability * (teacher - candidate)).sum())
        if forward_kl < -1e-8 or not math.isfinite(forward_kl):
            raise ValueError(f"invalid KL at position {served_position}: {forward_kl}")
        forward_kl = max(0.0, forward_kl)
        ref_top10 = set(teacher.topk(10).indices.tolist())
        candidate_top10 = set(candidate.topk(10).indices.tolist())
        per_position.append(
            {
                "served_position": served_position,
                "target_token_id": token_ids[served_position],
                "forward_kl": forward_kl,
                "top1_match": int(teacher.argmax() == candidate.argmax()),
                "top10_overlap": len(ref_top10 & candidate_top10) / 10.0,
            }
        )

    mean_kl = sum(float(item["forward_kl"]) for item in per_position) / positions
    max_kl = max(float(item["forward_kl"]) for item in per_position)
    top1 = sum(int(item["top1_match"]) for item in per_position) / positions
    top10 = sum(float(item["top10_overlap"]) for item in per_position) / positions
    report = {
        "kind": "fruit_full_vocabulary_kld",
        "schema_version": SCHEMA_VERSION,
        "reference": reference["metadata"],
        "reference_file": str(ref_path.resolve()),
        "reference_file_sha256": _sha256(ref_path),
        "prompt": reference["prompt"],
        "token_ids": token_ids,
        "served": str(served.resolve()),
        "served_hashes": _small_file_hashes(served),
        "kv_cache_dtype": os.environ.get("KV", "fp8_ds_mla"),
        "vllm_version": vllm.__version__,
        "positions": positions,
        "vocab_size": vocab_size,
        "mean_forward_kl": mean_kl,
        "max_forward_kl": max_kl,
        "top1_agreement": top1,
        "mean_top10_overlap": top10,
        "per_position": per_position,
    }
    output_path = os.environ.get("OUT_JSON")
    if output_path:
        Path(output_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    print("FRUIT-KLD " + json.dumps(report, sort_keys=True), flush=True)

    max_mean_kl = os.environ.get("MAX_MEAN_KL")
    min_top1 = os.environ.get("MIN_TOP1")
    if max_mean_kl is not None and mean_kl > float(max_mean_kl):
        raise SystemExit(
            f"FRUIT-KLD-FAIL mean {mean_kl:.6g} > {float(max_mean_kl):.6g}"
        )
    if min_top1 is not None and top1 < float(min_top1):
        raise SystemExit(f"FRUIT-KLD-FAIL top1 {top1:.6g} < {float(min_top1):.6g}")
    print("FRUIT-KLD-OK", flush=True)
    del output, prompt_logprobs, values, candidate, teacher, probability
    gc.collect()
    llm.llm_engine.engine_core.shutdown(timeout=30)
    del llm
    gc.collect()


def main() -> None:
    phase = os.environ.get("PHASE", "ref")
    prompt = os.environ.get("PROMPT", DEFAULT_PROMPT)
    ref_path = Path(os.environ.get("REF_PT", "/tmp/fruit_kld_ref.pt"))
    if phase == "ref":
        _reference_phase(prompt, ref_path)
    elif phase == "serve":
        _serve_phase(ref_path)
    else:
        raise ValueError(f"PHASE must be ref or serve, got {phase!r}")


if __name__ == "__main__":
    main()
