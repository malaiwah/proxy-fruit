#!/usr/bin/env python3
"""Round-trip parity: bf16 training graph vs SIQ-exported serving graph.

Phase 1 loads the training checkpoint through train_fruit's model classes
and records top-K next-token IDs + logprobs at each position of fixed
prompts. Phase 2 serves the export via vLLM (prompt_logprobs) and compares
top-1 agreement, top-K overlap, and a truncated top-K drift score. This score
is not a normalized KL divergence; use fruit_kld.py for full-vocabulary KLD.

Quantization introduces real divergence, so thresholds are loose; the
point is catching STRUCTURAL mismatches (wrong RoPE layout/theta reads as
near-zero top-1 agreement, not a few percent drift).

Env: CKPT (trainer .pt, raw state dict), SERVED (export dir), KV (dtype,
default fp8_ds_mla), TOPK (default 10), GEO_* as usual.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint_contract import checkpoint_conventions, checkpoint_model_state
import train_fruit as tf  # noqa: E402

PROMPTS = [
    "Once upon a time, there was a little girl named Lily. She loved",
    "MIT License\n\nCopyright (c) 2024 Example Corp\n\nPermission is hereby",
    "The quick brown fox jumps over the lazy dog. The capital of France",
]
K = int(os.environ.get("TOPK", "10"))

def _reference_conventions(state, environ):
    """Resolve trainer layout/theta without consuming model-state markers."""
    marker_state = {
        name: state[name]
        for name in ("serve_conv_v", "rope_theta_trained")
        if name in state
    }
    contract_env = dict(environ)
    if "FRUIT_ROPE_THETA" not in contract_env and "ROPE_THETA" in contract_env:
        contract_env["FRUIT_ROPE_THETA"] = contract_env["ROPE_THETA"]
    return checkpoint_conventions(marker_state, contract_env)


def main():
    dev = "cuda:0"
    phase = os.environ.get("PHASE", "ref")
    ref_pt = os.environ.get("REF_PT", "/tmp/parity_ref.pt")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.environ.get("TOK_DIR", "/mnt/vault/llm/glm52-franken/src"),
        trust_remote_code=False)
    enc = [tok.encode(p) for p in PROMPTS]

    if phase == "serve":
        payload = torch.load(ref_pt, weights_only=False)
        ref = payload["ref"]
        serve_phase(enc, ref)
        return

    print("[parity] phase 1: training graph", flush=True)
    sd = checkpoint_model_state(
        torch.load(os.environ["CKPT"], map_location="cpu", weights_only=False)
    )
    native, theta = _reference_conventions(sd, os.environ)
    tf.CONV["serve"] = native
    tf.THETA = theta
    print(f"[parity] checkpoint conventions: "
          f"{'serving' if native else 'legacy'}, theta={theta:g}", flush=True)
    model = tf.Fruit()
    model.load_state_dict(sd, strict=True)
    del sd
    model = model.to(dev, torch.bfloat16).eval()
    ref = []                       # per prompt: [T][K] ids + logprobs
    with torch.no_grad():
        for ids in enc:
            t = torch.tensor([ids], device=dev)
            h, _ = model(t)
            logits = model.lm_head(h[0]).float()
            lp = torch.log_softmax(logits, -1)
            top = lp.topk(K, dim=-1)
            ref.append((top.indices.cpu(), top.values.cpu()))
    del model
    torch.cuda.empty_cache()
    torch.save({"ref": ref}, ref_pt)
    print(f"[parity] reference written to {ref_pt}", flush=True)
    print("PARITY-REF-OK", flush=True)


def serve_phase(enc, ref):
    print("[parity] phase 2: served export", flush=True)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=os.environ["SERVED"], gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.75")),
              kv_cache_dtype=os.environ.get("KV", "fp8_ds_mla"),
              max_model_len=2048, max_num_seqs=4,
              max_num_batched_tokens=2048, enforce_eager=True,
              trust_remote_code=False)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=K)
    agree1 = overlap = kl_sum = n = 0
    for pi, ids in enumerate(enc):
        out = llm.generate([TokensPrompt(prompt_token_ids=list(ids))], sp,
                           use_tqdm=False)[0]
        plp = out.prompt_logprobs           # list, None at pos 0
        rid, rlp = ref[pi]
        # served position t gives dist for token t (predicted from <t);
        # trainer position t-1 predicts token t -> align t_served = t_ref+1
        for t in range(1, len(ids)):
            if plp[t] is None:
                continue
            served = {tid: l.logprob for tid, l in plp[t].items()}
            r_ids = rid[t - 1].tolist()
            r_lps = rlp[t - 1].tolist()
            s_top = max(served, key=served.get)
            agree1 += int(s_top == r_ids[0])
            overlap += len(set(r_ids) & set(served)) / K
            kl = 0.0
            for i_, tid in enumerate(r_ids):
                if tid in served:
                    kl += (2.718281828 ** r_lps[i_]) * \
                          (r_lps[i_] - served[tid])
            kl_sum += kl
            n += 1
    if n == 0:
        raise SystemExit("PARITY-STRUCTURAL-MISMATCH: no positions compared")
    agreement = agree1 / n
    print(f"PARITY: n={n} top1-agree={agreement*100:.1f}% "
          f"top{K}-overlap={overlap/n*100:.1f}% "
          f"mean-topK-drift={kl_sum/n:.4f}", flush=True)
    ok = agreement > 0.6          # structural-mismatch detector
    print("PARITY-" + ("OK" if ok else "STRUCTURAL-MISMATCH"), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
