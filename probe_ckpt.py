#!/usr/bin/env python3
"""Mid-run checkpoint probe (runs on the 5090, separate process from the
trainer): (1) proves the pushed checkpoint is loadable, (2) greedy-generates
to catch degenerate-output bugs early (TinyLlama's BOS-loop class),
(3) expert-load census + router logit stats per MoE layer (OLMoE: routing
fate is decided early), (4) embedding-norm glitch-token preview.

Env: CKPT (path to fruit_v1_main_ckpt.pt), GEO_* must match the run.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fruit as tf  # noqa: E402  (model classes + geometry from env)

CKPT = os.environ.get("CKPT", "/mnt/vault/llm/fruit-pilot/ckpt-probe/"
                              "checkpoints/fruit_v1_main_ckpt.pt")


def main():
    dev = "cuda:0"
    print(f"[probe] loading {CKPT}", flush=True)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    step = state.get("step", "?")
    sd = state["model"]
    del state
    model = tf.Fruit()
    model.load_state_dict(sd, strict=True)
    del sd
    model = model.to(dev, torch.bfloat16).eval()
    print(f"[probe] checkpoint step {step}: loads clean, "
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params",
          flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.environ.get("TOK_DIR", "/mnt/vault/llm/glm52-franken/src"),
        trust_remote_code=False)

    # --- router census hooks -------------------------------------------
    census = {}   # layer -> bincount over experts
    lstats = {}   # layer -> (logit_max, logit_mean)

    def make_hook(li):
        def hook(mod, args):
            x = args[0].reshape(-1, tf.H)
            with torch.no_grad():
                logits = mod.gate(x).float()
                sel = (torch.sigmoid(logits)
                       + mod.e_score_correction_bias).topk(tf.TOPK, -1).indices
                c = torch.bincount(sel.reshape(-1), minlength=tf.N_EXP)
                census[li] = census.get(li, 0) + c.cpu()
                mx, mn = float(logits.max()), float(logits.mean())
                pm = lstats.get(li, (-1e9, 0.0))
                lstats[li] = (max(pm[0], mx), mn)
        return hook

    hooks = [blk.mlp.register_forward_pre_hook(make_hook(i))
             for i, blk in enumerate(model.layers) if blk.is_moe]

    # --- generation probes ---------------------------------------------
    prompts = ["Once upon a time",
               "MIT License\n\nCopyright (c)",
               "The capital of France is",
               "def fibonacci(n):",
               "人工智能是"]
    with torch.no_grad():
        for prompt in prompts:
            ids = torch.tensor([tok.encode(prompt)], device=dev)
            for _ in range(60):
                h, _ = model(ids[:, -4096:])
                nxt = model.lm_head(h[:, -1:]).argmax(-1)
                ids = torch.cat([ids, nxt], dim=1)
            out = tok.decode(ids[0])[len(prompt):]
            # degenerate-output tell: unique-token ratio of the continuation
            cont = ids[0, -60:].tolist()
            uniq = len(set(cont)) / len(cont)
            print(f"[gen] {prompt!r} (uniq {uniq:.2f})\n      -> {out[:160]!r}",
                  flush=True)

    for h in hooks:
        h.remove()

    # --- census report --------------------------------------------------
    print("[router] per-layer expert load (census over the gens above):",
          flush=True)
    for li in sorted(census):
        c = census[li].float()
        frac = c / c.sum()
        dead = int((frac < 0.1 / tf.N_EXP).sum())   # <10% of uniform share
        near0 = int((c == 0).sum())
        top = float(frac.max())
        ent = float(-(frac.clamp_min(1e-12) * frac.clamp_min(1e-12).log())
                    .sum() / torch.log(torch.tensor(float(tf.N_EXP))))
        mx, mn = lstats[li]
        print(f"  layer {li:2d}: entropy {ent:.3f} (1=uniform)  "
              f"unused {near0:3d}/{tf.N_EXP}  starved(<10% uni) {dead:3d}  "
              f"maxload {top*100:.1f}%  logit max {mx:.1f} mean {mn:.2f}",
              flush=True)

    # --- glitch-token preview -------------------------------------------
    with torch.no_grad():
        en = model.embed_tokens.weight.float().norm(dim=1)
        hn = model.lm_head.weight.float().norm(dim=1)
        med = en.median()
        tiny = int((en < 0.3 * med).sum())
        print(f"[glitch] embed-norm median {med:.3f}; tokens <30% of median: "
              f"{tiny}/{tf.VOCAB} ({tiny/tf.VOCAB*100:.1f}%) — "
              f"undertrained-vocab preview; head-norm median {hn.median():.3f}",
              flush=True)
    print("PROBE-DONE", flush=True)


if __name__ == "__main__":
    main()
