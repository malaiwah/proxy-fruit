#!/usr/bin/env python3
"""GLM-5.2-SIQ-Fruit PILOT: train a micro GLM-5.2-shaped MoE from scratch.

Faithful structure, tiny sizes: MLA (kv_lora 512, rope 64, head dims
192/256 kept), sigmoid noaux_tc router + e_score_correction_bias (zeros,
selection-compatible), 256 routed experts top-8 + 1 shared, first 3 layers
dense, one MTP draft layer. DSA indexer weights are carried (random init,
never trained): at serving seqs <= index_topk the indexer selects
everything, so only its shapes matter.

Corpus: TinyStories slice + oversampled license corpus.
Output: BF16 state dict + config, consumed by export_fruit.py.
"""
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("FRUIT_OUT_DIR", "/mnt/vault/llm/fruit-pilot"))
SRC = Path("/mnt/vault/llm/glm52-franken/src")
CORPUS = Path("/mnt/vault/llm/glm52-franken/corpus")

# ---- pilot geometry (scale of GLM-5.2; serving-critical dims KEPT) ----
_e = os.environ.get
H = int(_e("GEO_H", "512"))               # hidden (parent 6144)
NL = int(_e("GEO_NL", "6"))               # dense + MoE (+1 MTP appended)
FIRST_MOE = 3
HEADS = int(_e("GEO_HEADS", "8"))         # >=8: SM120 sparse-MLA dispatch
QK_NOPE, QK_ROPE, V_HEAD = 192, 64, 256   # KEPT (kernel parity)
KV_LORA = 512                             # KEPT
Q_LORA = int(_e("GEO_QLORA", "512"))
DENSE_INTER = int(_e("GEO_DENSE_INTER", "1024"))
MOE_INTER = int(_e("GEO_MOE_INTER", "128"))  # >=256 for uniform-tier path
N_EXP, TOPK, N_SHARED = 256, 8, 1
ROUTED_SCALE = 2.5
VOCAB = 154880
IDX_HEADS, IDX_DIM = 32, 128              # KEPT
SEQ = int(os.environ.get("SEQ", "512"))
THETA = float(os.environ.get("ROPE_THETA", "10000"))
RESUME_PT = os.environ.get("RESUME_PT", "")
DISTILL = os.environ.get("INDEXER_DISTILL", "") == "1"
GRAD_CKPT = os.environ.get("GRAD_CKPT", "") == "1"
SAVE_NAME = os.environ.get("SAVE_NAME", "fruit_pilot")
LR = float(os.environ.get("LR", "6e-4"))
MTP_W = float(os.environ.get("MTP_W", "0.3"))   # DSv3: 0.3 early, 0.1 late/SFT
# ---- run-2 kit (all default-off; run-1 recipe unchanged) ----
MTP_W_END = float(os.environ.get("MTP_W_END", "0") or 0)  # >0: linear decay
AUX_COEF = float(os.environ.get("AUX_COEF", "0.003"))
BIAS_BALANCE = float(os.environ.get("BIAS_BALANCE", "0"))  # DSv3 gamma
ZLOSS_HEAD = float(os.environ.get("ZLOSS_HEAD", "0"))      # OLMo2 z-loss
TIE_EMB = os.environ.get("TIE_EMB", "") == "1"
NO_WD_EMB = os.environ.get("NO_WD_EMB", "") == "1"
LR_SCHED = os.environ.get("LR_SCHED", "cosine")            # cosine | wsd
WSD_DECAY_FRAC = float(os.environ.get("WSD_DECAY_FRAC", "0.2"))
WARMUP = int(os.environ.get("WARMUP", "100"))
INTRADOC_MASK = os.environ.get("INTRADOC_MASK", "") == "1"
DET_DATA = os.environ.get("DETERMINISTIC_DATA", "") == "1"
# Token-indexed training: schedule/budget as f(tokens_seen) instead of
# f(step) -> resume is INSTANCE-TIER-AGNOSTIC (any NPROC x BS continues the
# LR curve exactly). TOKENS_SEEN_INIT migrates a step-indexed ckpt once
# (tokens = step x old_tokens_per_step).
TOKEN_BUDGET = int(float(os.environ.get("TOKEN_BUDGET", "0") or 0))
TOKENS_SEEN_INIT = int(float(os.environ.get("TOKENS_SEEN_INIT", "0") or 0))
# QAT-lite: Gaussian weight noise at trellis-quant-error scale during the
# anneal tail -> flat-minima w.r.t. the perturbation SIQ will apply.
# Relative sigma: K4 trellis error ~ 1-2% of weight std -> QNOISE=0.015.
QNOISE = float(os.environ.get("QNOISE", "0"))
EOS_ID = int(os.environ.get("EOS_ID", "154820"))           # <|endoftext|>


class _FP8MM(torch.autograd.Function):
    """Tensorwise-dynamic-scaled FP8 matmul (fwd + dgrad + wgrad all in
    e4m3 via torch._scaled_mm). Measured 2.1-3.3x vs bf16 at Fruit shapes
    on sm120 (5090). Enable per-layer via FP8_LINEAR=1."""

    @staticmethod
    def forward(ctx, x, w):
        xf = x.reshape(-1, x.shape[-1])
        sx = (xf.abs().amax().float().clamp(min=1e-4) / 448.0)
        sw = (w.abs().amax().float().clamp(min=1e-4) / 448.0)
        x8 = (xf / sx).to(torch.float8_e4m3fn)
        w8 = (w / sw).to(torch.float8_e4m3fn)
        ctx.save_for_backward(x8, w8, sx, sw)
        y = torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw,
                             out_dtype=torch.bfloat16)
        return y.reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x8, w8, sx, sw = ctx.saved_tensors
        g = gy.reshape(-1, gy.shape[-1])
        sg = (g.abs().amax().float().clamp(min=1e-4) / 448.0)
        g8 = (g / sg).to(torch.float8_e4m3fn)
        w8c = w8.t().contiguous()            # [K,N] row-major
        gx = torch._scaled_mm(g8, w8c.t(), scale_a=sg, scale_b=sw,
                              out_dtype=torch.bfloat16)
        g8c = g8.t().contiguous()            # [N,M] row-major
        gw = torch._scaled_mm(g8c, x8.t().contiguous().t(), scale_a=sg,
                              scale_b=sx, out_dtype=torch.bfloat16)
        return gx.reshape(gy.shape[:-1] + (w8.shape[1],)), gw


def fp8ify(model, min_dim=512):
    """Route eligible nn.Linear forwards through _FP8MM (weights stay bf16
    masters; FP8 is the compute format only — serving parity unaffected)."""
    n = 0
    for name, mod in model.named_modules():
        # mtp path sees BS*(SEQ-1) tokens — trailing-dim%16 violates
        # _scaled_mm's wgrad requirement; main path is always divisible
        if isinstance(mod, nn.Linear) and mod.bias is None \
                and "mtp" not in name \
                and min(mod.in_features, mod.out_features) >= min_dim \
                and mod.out_features < 100_000:      # skip lm_head
            mod.forward = (lambda x, m=mod: _FP8MM.apply(x, m.weight))
            n += 1
    print(f"[fp8] {n} linears routed through _scaled_mm", flush=True)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5)
        return (v * self.weight.float()).to(x.dtype)


def rope(x, pos, theta=None):
    theta = THETA if theta is None else theta
    # non-interleaved rotate-half on the last dim (DeepSeek convention)
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]
    sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (x.float() * cos + rot.float() * sin).to(x.dtype)


class MLA(nn.Module):
    def __init__(self):
        super().__init__()
        qk_head = QK_NOPE + QK_ROPE
        self.q_a_proj = nn.Linear(H, Q_LORA, bias=False)
        self.q_a_layernorm = RMSNorm(Q_LORA)
        self.q_b_proj = nn.Linear(Q_LORA, HEADS * qk_head, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(H, KV_LORA + QK_ROPE, bias=False)
        self.kv_a_layernorm = RMSNorm(KV_LORA)
        self.kv_b_proj = nn.Linear(KV_LORA, HEADS * (QK_NOPE + V_HEAD), bias=False)
        self.o_proj = nn.Linear(HEADS * V_HEAD, H, bias=False)

    def forward(self, x, pos, doc_mask=None):
        B, T, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(B, T, HEADS, QK_NOPE + QK_ROPE)
        q_nope, q_rope = q.split([QK_NOPE, QK_ROPE], dim=-1)
        q_rope = rope(q_rope, pos)
        kv_a = self.kv_a_proj_with_mqa(x)
        c_kv, k_rope = kv_a.split([KV_LORA, QK_ROPE], dim=-1)
        k_rope = rope(k_rope.unsqueeze(2), pos).expand(B, T, HEADS, QK_ROPE)
        kv = self.kv_b_proj(self.kv_a_layernorm(c_kv))
        kv = kv.view(B, T, HEADS, QK_NOPE + V_HEAD)
        k_nope, v = kv.split([QK_NOPE, V_HEAD], dim=-1)
        qh = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        kh = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        vh = v.transpose(1, 2)
        if doc_mask is not None:        # block-diagonal: causal AND same-doc
            o = F.scaled_dot_product_attention(qh, kh, vh,
                                               attn_mask=doc_mask)
        else:
            o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        self.distill_loss = None
        if DISTILL and T > 64:
            # Teach the indexer to mimic full attention (DSv3.2 recipe):
            # I[t,s] = sum_h w[t,h] * relu(qi[t,h] . ki[s]); KL to the
            # head-mean attention row at 32 sampled query positions.
            with torch.no_grad():
                qs = torch.randint(T // 2, T, (32,), device=x.device)
                logits = torch.einsum("bqhd,bshd->bqhs",
                                      qh.transpose(1, 2)[:, qs].float(),
                                      kh.transpose(1, 2).float())
                logits = logits / math.sqrt(QK_NOPE + QK_ROPE)
                mask = (torch.arange(T, device=x.device)[None, :]
                        > qs[:, None])
                logits = logits.masked_fill(mask[None, :, None, :],
                                            float("-inf"))
                target = logits.softmax(-1).mean(2)      # [B, 32, T]
            idx = self.indexer
            xn = x.detach()
            ki = idx.k_norm(idx.wk(xn))                  # [B, T, 128]
            ki = rope(ki.unsqueeze(2), pos).squeeze(2)   # serving parity:
            qi = idx.wq_b(self.q_a_layernorm(            # indexer q/k are
                self.q_a_proj(xn))).view(B, T, IDX_HEADS, IDX_DIM)
            qi = rope(qi, pos)                           # rotated in GLM
            wi = idx.weights_proj(xn)                    # [B, T, H_idx]
            scores = torch.einsum("bqhd,bsd->bqhs",
                                  qi[:, qs].float(), ki.float()).relu()
            iscore = torch.einsum("bqh,bqhs->bqs",
                                  wi[:, qs].float(), scores)
            # finite mask fill: exact -inf makes kl_div emit 0*(-inf)=nan
            iscore = iscore.masked_fill(mask[None], -1e9)
            kl = F.kl_div(iscore.log_softmax(-1), target,
                          reduction="batchmean")
            self.distill_loss = kl if torch.isfinite(kl) else None
        return self.o_proj(o.transpose(1, 2).reshape(B, T, HEADS * V_HEAD))


class Indexer(nn.Module):
    """DSA indexer: parameters only (never in the training graph)."""

    def __init__(self):
        super().__init__()
        self.wk = nn.Linear(H, IDX_DIM, bias=False)
        self.wq_b = nn.Linear(Q_LORA, IDX_HEADS * IDX_DIM, bias=False)
        self.weights_proj = nn.Linear(H, IDX_HEADS, bias=False)
        self.k_norm = nn.LayerNorm(IDX_DIM)
        if not DISTILL:
            for p in self.parameters():
                p.requires_grad_(False)


class DenseMLP(nn.Module):
    def __init__(self, inter):
        super().__init__()
        self.gate_proj = nn.Linear(H, inter, bias=False)
        self.up_proj = nn.Linear(H, inter, bias=False)
        self.down_proj = nn.Linear(inter, H, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, N_EXP, bias=False)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(N_EXP), requires_grad=False)
        self.experts = nn.ModuleList(DenseMLP(MOE_INTER) for _ in range(N_EXP))
        self.shared_experts = DenseMLP(MOE_INTER * N_SHARED)

    def forward(self, x):
        B, T, _ = x.shape
        flat = x.reshape(-1, H)
        scores = torch.sigmoid(self.gate(flat).float())
        sel = (scores + self.e_score_correction_bias).topk(TOPK, dim=-1).indices
        w = scores.gather(-1, sel)
        w = (w / w.sum(-1, keepdim=True)) * ROUTED_SCALE
        out = torch.zeros_like(flat)
        for e in range(N_EXP):
            mask = (sel == e)
            rows = mask.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            ww = (w * mask.float()).sum(-1)[rows].unsqueeze(-1)
            out.index_add_(0, rows,
                           (self.experts[e](flat[rows]) * ww).to(out.dtype))
        # aux load-balance loss (mean fraction * mean prob, DSv2-style)
        me = scores.mean(0)
        ce = torch.zeros(N_EXP, device=x.device)
        ce.scatter_add_(0, sel.reshape(-1),
                        torch.ones(sel.numel(), device=x.device))
        ce = ce / sel.numel()
        self.aux = (me * ce).sum() * N_EXP
        return (out + self.shared_experts(flat)).view(B, T, H)


class StackedMoE(nn.Module):
    """Grouped-GEMM MoE: experts stacked into three [E, ...] tensors, tokens
    sorted by expert and run as padded bmm — one launch per projection
    instead of one per expert. Pure torch; works on any CUDA arch.
    Enable with MOE_IMPL=stacked."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, N_EXP, bias=False)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(N_EXP), requires_grad=False)
        s = 1.0 / math.sqrt(H)
        self.w_gate = nn.Parameter(torch.randn(N_EXP, MOE_INTER, H) * s)
        self.w_up = nn.Parameter(torch.randn(N_EXP, MOE_INTER, H) * s)
        self.w_down = nn.Parameter(
            torch.randn(N_EXP, H, MOE_INTER) / math.sqrt(MOE_INTER))
        self.shared_experts = DenseMLP(MOE_INTER * N_SHARED)

    def forward(self, x):
        B, T, _ = x.shape
        flat = x.reshape(-1, H)
        N = flat.shape[0]
        logits = self.gate(flat).float()
        self.zloss = (logits ** 2).mean()   # ST-MoE-style logit bound
        scores = torch.sigmoid(logits)
        sel = (scores + self.e_score_correction_bias).topk(TOPK, -1).indices
        w = scores.gather(-1, sel)
        w = (w / w.sum(-1, keepdim=True)) * ROUTED_SCALE
        sel_flat = sel.reshape(-1)
        order = sel_flat.argsort(stable=True)
        counts = torch.bincount(sel_flat, minlength=N_EXP)
        self.last_counts = counts.detach()   # for aux-free bias balancing
        M = int(counts.max().item())
        # position of each (token,slot) inside its expert's group
        offs = counts.cumsum(0) - counts
        slot_in_grp = (torch.arange(N * TOPK, device=x.device)
                       - offs.repeat_interleave(counts))
        grp_expert = sel_flat[order]
        grp_tok = order // TOPK
        G = flat.new_zeros(N_EXP, M, H)
        G[grp_expert, slot_in_grp] = flat[grp_tok]
        h1 = F.silu(torch.bmm(G, self.w_gate.transpose(1, 2)))
        h1 = h1 * torch.bmm(G, self.w_up.transpose(1, 2))
        Y = torch.bmm(h1, self.w_down.transpose(1, 2))   # [E, M, H]
        y_sorted = Y[grp_expert, slot_in_grp]            # [N*TOPK, H]
        ww = w.reshape(-1)[order].unsqueeze(-1)
        out = torch.zeros_like(flat)
        out.index_add_(0, grp_tok, (y_sorted * ww).to(out.dtype))
        me = scores.mean(0)
        ce = counts.float() / sel_flat.numel()
        self.aux = (me * ce).sum() * N_EXP
        return (out + self.shared_experts(flat)).view(B, T, H)


class GroupedMoE(StackedMoE):
    """StackedMoE with torch._grouped_mm (torch>=2.7, differentiable):
    ragged expert groups, NO padding to max-count — the training-grade
    version of the inference-stack fused-MoE dispatch. MOE_IMPL=grouped."""

    def forward(self, x):
        B, T, _ = x.shape
        flat = x.reshape(-1, H)
        logits = self.gate(flat).float()
        self.zloss = (logits ** 2).mean()
        scores = torch.sigmoid(logits)
        sel = (scores + self.e_score_correction_bias).topk(TOPK, -1).indices
        w = scores.gather(-1, sel)
        w = (w / w.sum(-1, keepdim=True)) * ROUTED_SCALE
        sel_flat = sel.reshape(-1)
        order = sel_flat.argsort(stable=True)
        counts = torch.bincount(sel_flat, minlength=N_EXP)
        self.last_counts = counts.detach()
        grp_tok = order // TOPK
        xs = flat[grp_tok]                          # [N*TOPK, H] sorted
        offs = counts.cumsum(0, dtype=torch.int32)  # group end offsets
        h1 = F.silu(torch._grouped_mm(xs, self.w_gate.transpose(1, 2),
                                      offs=offs))
        h1 = h1 * torch._grouped_mm(xs, self.w_up.transpose(1, 2), offs=offs)
        y = torch._grouped_mm(h1, self.w_down.transpose(1, 2), offs=offs)
        ww = w.reshape(-1)[order].unsqueeze(-1)
        out = torch.zeros_like(flat)
        out.index_add_(0, grp_tok, (y * ww).to(out.dtype))
        me = scores.mean(0)
        ce = counts.float() / sel_flat.numel()
        self.aux = (me * ce).sum() * N_EXP
        return (out + self.shared_experts(flat)).view(B, T, H)


MOE_CLS = {"stacked": StackedMoE, "grouped": GroupedMoE}.get(
    os.environ.get("MOE_IMPL", ""), MoE)


class Block(nn.Module):
    def __init__(self, li, moe):
        super().__init__()
        self.input_layernorm = RMSNorm(H)
        self.self_attn = MLA()
        self.self_attn.indexer = Indexer()
        self.post_attention_layernorm = RMSNorm(H)
        self.mlp = MOE_CLS() if moe else DenseMLP(DENSE_INTER)
        self.is_moe = moe

    def forward(self, x, pos, doc_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), pos, doc_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Fruit(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, H)
        self.layers = nn.ModuleList(
            Block(i, i >= FIRST_MOE) for i in range(NL))
        self.norm = RMSNorm(H)
        self.lm_head = nn.Linear(H, VOCAB, bias=False)
        if TIE_EMB:                     # export writes both copies anyway
            # tying REQUIRES small init: default N(0,1) embeddings become
            # the output head -> logits x30 -> loss ~380 at step 0 (measured)
            nn.init.normal_(self.embed_tokens.weight, std=0.02)
            self.lm_head.weight = self.embed_tokens.weight
        # MTP draft layer (predicts t+2): enorm/hnorm/eh_proj + one block
        self.mtp_enorm = RMSNorm(H)
        self.mtp_hnorm = RMSNorm(H)
        self.mtp_eh_proj = nn.Linear(2 * H, H, bias=False)
        self.mtp_block = Block(NL, True)

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.embed_tokens(ids)
        dm = None
        if INTRADOC_MASK:
            T = ids.shape[1]
            # doc id increments AFTER each EOS (EOS stays with its doc)
            b = torch.zeros_like(ids)
            b[:, 1:] = (ids[:, :-1] == EOS_ID).long()
            doc = b.cumsum(1)
            same = doc[:, :, None] == doc[:, None, :]
            causal = torch.ones(T, T, dtype=torch.bool,
                                device=ids.device).tril()
            dm = (same & causal)[:, None]        # [B,1,T,T] bool
        for blk in self.layers:
            if GRAD_CKPT and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(blk, x, pos, dm, use_reentrant=False)
            else:
                x = blk(x, pos, dm)
        h = self.norm(x)
        # MTP: combine hidden(t) with embed(t+1) -> predict t+2
        # CONVENTION CONTRACT: this cat order ([hidden, embed]) and the
        # half-split RoPE above are the TRAINING conventions; vLLM serves
        # the opposite ([embed, hidden] eh_proj, interleaved RoPE) and
        # export_fruit.py converts BOTH at export time. Do not "fix"
        # either side alone — flipping here without removing the export
        # conversion double-converts and silently breaks serving
        # (measured: MTP acceptance 98% -> 0.4%).
        emb_next = self.embed_tokens(ids[:, 1:])
        mtp_in = self.mtp_eh_proj(torch.cat(
            [self.mtp_hnorm(x[:, :-1]), self.mtp_enorm(emb_next)], dim=-1))
        dm_m = dm[:, :, :-1, :-1] if dm is not None else None
        if GRAD_CKPT and self.training:
            from torch.utils.checkpoint import checkpoint
            mtp_h = checkpoint(self.mtp_block, mtp_in, pos[:-1], dm_m,
                               use_reentrant=False)
        else:
            mtp_h = self.mtp_block(mtp_in, pos[:-1], dm_m)
        return h, self.norm(mtp_h)


def ce_chunked(head, h, tgt, chunk=None):
    """Cross-entropy without materializing [T, vocab]: head+CE per chunk,
    recomputed in backward. Chunk adapts to batch size so the fp32 logits
    transient stays ~1.3 GiB regardless of BS (10 GiB at BS=8 otherwise)."""
    from torch.utils.checkpoint import checkpoint
    if chunk is None:
        chunk = max(128, 2048 // max(1, h.shape[0]))

    def piece(hc, tc):
        logits = head(hc).float().reshape(-1, VOCAB)
        tcf = tc.reshape(-1)
        ce = F.cross_entropy(logits, tcf, reduction="sum",
                             ignore_index=-100)
        if ZLOSS_HEAD > 0:            # OLMo2: z = logsumexp^2, valid rows
            z = torch.logsumexp(logits[tcf != -100], -1).pow(2).sum()
            return ce + ZLOSS_HEAD * z
        return ce

    total = 0
    loss = None
    for i in range(0, h.shape[1], chunk):
        p = checkpoint(piece, h[:, i:i + chunk], tgt[:, i:i + chunk],
                       use_reentrant=False)
        loss = p if loss is None else loss + p
        total += int((tgt[:, i:i + chunk] != -100).sum())
    if total == 0:                    # window with no loss-bearing tokens:
        return loss * 0.0             # keep head in the graph (DDP shape)
    return loss / total


class ShardMix:
    """Memmap uint32 shards + weighted source sampling; the last val_tokens
    of every shard are excluded from training and provide fixed val windows."""

    def __init__(self, shard_dir):
        m = json.loads((Path(shard_dir) / "manifest.json").read_text())
        self.names = [n for n in m["weights"] if m["tokens"][n] > 0]
        w = np.array([m["weights"][n] for n in self.names], dtype=np.float64)
        self.weights = w / w.sum()
        self.val_tokens = int(m.get("val_tokens", 262144))
        self.maps = {n: np.memmap(Path(shard_dir) / f"{n}.u32",
                                  dtype=np.uint32, mode="r")
                     for n in self.names}
        # SFT mode: parallel .mask.u8 shards (1 = token carries loss).
        # Per-source: a source WITHOUT a mask file trains on all tokens —
        # that is the pretrain-replay channel (anti-forgetting).
        self.masks = {n: np.memmap(Path(shard_dir) / f"{n}.mask.u8",
                                   dtype=np.uint8, mode="r")
                      for n in self.names
                      if (Path(shard_dir) / f"{n}.mask.u8").exists()}
        if not self.masks:
            self.masks = None
        else:
            print(f"[shards] SFT loss masks active "
                  f"({len(self.masks)}/{len(self.names)} sources; "
                  "rest = full-loss replay)", flush=True)
        # conv-aligned sampling: windows start at conversation boundaries
        # (random cuts orphan assistant tokens from their prompts)
        self.starts = {n: np.memmap(Path(shard_dir) / f"{n}.starts.u64",
                                    dtype=np.uint64, mode="r")
                       for n in self.names
                       if (Path(shard_dir) / f"{n}.starts.u64").exists()}
        self.consumed = {n: 0 for n in self.names}
        print(f"[shards] {len(self.names)} sources, "
              f"{sum(len(v) for v in self.maps.values())/1e9:.2f}B tokens",
              flush=True)

    def batch(self, bs, seq, rng):
        out = np.empty((bs, seq + 2), dtype=np.int64)
        msk = np.empty((bs, seq + 2), dtype=np.int64) \
            if self.masks is not None else None
        for i in range(bs):
            n = self.names[rng.choice(len(self.names), p=self.weights)]
            self.consumed[n] += 1
            arr = self.maps[n]
            hi = len(arr) - self.val_tokens - seq - 2
            j = -1
            if n in self.starts:
                s = self.starts[n]
                j = int(s[int(rng.integers(0, len(s)))])
            if not 0 <= j <= hi:
                j = int(rng.integers(0, max(hi, 1)))
            out[i] = arr[j:j + seq + 2]
            if msk is not None:
                msk[i] = self.masks[n][j:j + seq + 2] \
                    if n in self.masks else 1
        return torch.from_numpy(out), \
            (torch.from_numpy(msk) if msk is not None else None)

    def val_windows(self, seq, per_source=2):
        for n in self.names:
            arr = self.maps[n]
            base = len(arr) - self.val_tokens
            for k in range(per_source):
                j = base + k * (seq + 2)
                if j + seq + 2 <= len(arr):
                    m = None
                    if self.masks and n in self.masks:
                        m = torch.from_numpy(np.asarray(
                            self.masks[n][j:j + seq + 2], dtype=np.int64))
                    yield n, torch.from_numpy(
                        np.asarray(arr[j:j + seq + 2], dtype=np.int64)), m


import numpy as np


def plot_val_progress(rec, out_dir, save_name, push_repo=None):
    """PLOT_VAL=1 (Run-2): the trainer persists each val sweep to
    {save_name}_val.jsonl and renders the val-progress plot itself — no
    external log parsing. Resume-safe: the jsonl is append-only and the
    plot dedups by step (last write wins). Small files; pushed to the HF
    ckpt repo's progress/ dir when pushing is configured."""
    import threading
    hist_path = out_dir / f"{save_name}_val.jsonl"
    with open(hist_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    rows = {}
    for line in open(hist_path):
        try:
            r = json.loads(line)
            rows[r["step"]] = r
        except Exception:
            continue
    rows = [rows[s] for s in sorted(rows)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    meta = ("step", "tokens_seen", "wall", "global")
    for k in sorted({k for r in rows for k in r if k not in meta}):
        pts = [(r["step"], r[k]) for r in rows if k in r]
        ax.plot(*zip(*pts), lw=0.9, alpha=0.7, label=k)
    ax.plot([r["step"] for r in rows], [r["global"] for r in rows],
            "k-", lw=2.2, label="global")
    ax.set_xlabel(f"step  ({rows[-1].get('tokens_seen', 0)/1e9:.2f}B "
                  "tokens seen)")
    ax.set_ylabel("val loss")
    ax.set_title(f"{save_name} val progress")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    png = out_dir / f"{save_name}_val.png"
    fig.savefig(str(png) + ".tmp.png", dpi=110)
    plt.close(fig)
    os.replace(str(png) + ".tmp.png", png)
    print(f"[plot-val] {png.name}: {len(rows)} sweeps", flush=True)
    if push_repo:
        def up():
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=os.environ["HF_TOKEN"])
                for p in (png, hist_path):
                    api.upload_file(
                        path_or_fileobj=str(p),
                        path_in_repo=f"progress/{p.name}",
                        repo_id=push_repo,
                        commit_message=f"val progress step {rec['step']}")
            except Exception as exc:
                print(f"[plot-val] push failed: {exc}", flush=True)
        threading.Thread(target=up, daemon=True).start()


def load_data(tok):
    texts = []
    for p in sorted(CORPUS.glob("*.txt")):
        texts.append(p.read_text())
    lic = "\n\n".join(texts)
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train[:40000]")
        stories = "\n\n".join(ds["text"])
    except Exception as e:
        print(f"[data] TinyStories unavailable ({e}); licenses only", flush=True)
        stories = ""
    # REAP recall calibration set (brandonmusic GLM-5.2 toolchain): the
    # "general use" distribution the real encoder calibrates against.
    calib_txts = []
    calib_path = Path("/mnt/vault/llm/fruit-pilot/reap_recall_calib.jsonl")
    if calib_path.exists():
        for line in calib_path.open():
            t = json.loads(line).get("text", "")
            try:  # rows wrap serialized chats; extract the turns as prose
                msgs = json.loads(t).get("messages", [])
                t = "\n".join(m.get("content", "") for m in msgs)
            except Exception:
                pass
            calib_txts.append(t)
    calib = "\n\n".join(calib_txts)
    ids = (tok.encode(stories) + tok.encode(calib)
           + tok.encode(lic) * 25)   # oversample licenses
    print(f"[data] {len(ids)/1e6:.1f}M tokens (stories + "
          f"{len(calib_txts)} calib rows + "
          f"{len(tok.encode(lic))/1e3:.0f}k license x25)", flush=True)
    return torch.tensor(ids, dtype=torch.long)


def main():
    # torchrun DDP support: RANK/WORLD_SIZE env → multi-GPU data parallel
    ddp = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.manual_seed(20260805 + rank)
    if ddp:
        torch.distributed.init_process_group("nccl")
        dev = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(dev)
    else:
        dev = "cuda:0"
    is_main = rank == 0
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.environ.get("TOK_DIR", str(SRC)), trust_remote_code=False)
    shard_dir = os.environ.get("SHARD_DIR", "")
    mix = ShardMix(shard_dir) if shard_dir else None
    data = None if mix else load_data(tok)
    rng = np.random.default_rng(20260805 + rank)
    push_repo = os.environ.get("HF_PUSH_REPO", "")
    push_every = int(os.environ.get("PUSH_EVERY", "1500"))
    eval_every = int(os.environ.get("EVAL_EVERY", "1000"))
    import threading
    push_state = {"t": None}
    snap_state = {"mirror": None, "saver": None}

    def _snap_tree(src_obj, mirror):
        """Recursively copy tensors into pinned mirrors (non_blocking) and
        rebuild non-tensor leaves fresh. Returns the snapshot object."""
        if torch.is_tensor(src_obj):
            if mirror is None or mirror.shape != src_obj.shape \
                    or mirror.dtype != src_obj.dtype:
                mirror = torch.empty_like(src_obj, device="cpu",
                                          pin_memory=True)
            mirror.copy_(src_obj, non_blocking=True)
            return mirror
        if isinstance(src_obj, dict):
            if not isinstance(mirror, dict):
                mirror = {}
            return {k: _snap_tree(v, mirror.get(k))
                    for k, v in src_obj.items()}
        if isinstance(src_obj, (list, tuple)):
            if not isinstance(mirror, list):
                mirror = [None] * len(src_obj)
            out = [_snap_tree(v, mirror[i] if i < len(mirror) else None)
                   for i, v in enumerate(src_obj)]
            return out if isinstance(src_obj, list) else tuple(out)
        return src_obj                     # scalars/None copied by value

    def push_ckpt(step):
        if not push_repo:
            return
        if push_state["t"] and push_state["t"].is_alive():
            return
        def go():
            # hardlink snapshot: os.replace on ckpt_path swaps the inode,
            # so the upload keeps reading a consistent file even if a
            # newer save lands mid-upload
            link = str(ckpt_path) + ".push"
            try:
                try:
                    os.unlink(link)
                except FileNotFoundError:
                    pass
                os.link(ckpt_path, link)
                from huggingface_hub import HfApi
                HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                    path_or_fileobj=link,
                    path_in_repo=f"checkpoints/{SAVE_NAME}_ckpt.pt",
                    repo_id=push_repo, commit_message=f"step {step}")
                print(f"[push] step {step} -> {push_repo}", flush=True)
            except Exception as exc:
                print(f"[push] failed: {exc}", flush=True)
            finally:
                try:
                    os.unlink(link)
                except FileNotFoundError:
                    pass
        push_state["t"] = threading.Thread(target=go, daemon=True)
        push_state["t"].start()

    def run_val(step):
        if mix is None:
            return
        model.eval()
        with torch.no_grad():
            per = {}
            for nname, w, wm in mix.val_windows(min(SEQ, 4096)):
                w = w.to(dev).unsqueeze(0)
                hh, _ = model(w[:, :-2])
                vt = w[:, 1:-1]
                if wm is not None:      # SFT: assistant-token val loss only
                    vt = vt.masked_fill(
                        wm.to(dev).unsqueeze(0)[:, 1:-1] == 0, -100)
                l = ce_chunked(raw_model.lm_head, hh, vt)
                per.setdefault(nname, []).append(float(l))
            msg = "  ".join(f"{k}={sum(v)/len(v):.3f}"
                            for k, v in sorted(per.items()))
            gl = sum(sum(v) for v in per.values()) / sum(
                len(v) for v in per.values())
            print(f"[val {step}] global={gl:.4f}  {msg}", flush=True)
            tot = max(1, sum(mix.consumed.values()))
            print("[mix " + str(step) + "] " + "  ".join(
                f"{k}={v/tot:.3f}" for k, v in
                sorted(mix.consumed.items())), flush=True)
        if os.environ.get("PLOT_VAL", "") == "1":
            rec = {"step": step, "tokens_seen": tok_state["seen"],
                   "wall": round(time.time(), 1), "global": round(gl, 5)}
            rec.update({k: round(sum(v) / len(v), 5)
                        for k, v in per.items()})
            try:
                plot_val_progress(rec, OUT, SAVE_NAME, push_repo)
            except Exception as exc:
                print(f"[plot-val] failed: {exc}", flush=True)
        model.train()
    model = Fruit().to(dev, torch.bfloat16)
    if os.environ.get("FP8_LINEAR", "") == "1":
        fp8ify(model)
    raw_model = model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # MoE: routed experts make per-step-varying grad sets
        model = DDP(model, device_ids=[torch.cuda.current_device()],
                    static_graph=True, gradient_as_bucket_view=True)
        print(f"[ddp] rank {rank}/{world}", flush=True)
    if os.environ.get("COMPILE", "") == "1":
        try:
            model = torch.compile(model)
            print("[compile] torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"[compile] failed, eager: {exc}", flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_par/1e6:.0f}M params", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    masters = None
    if os.environ.get("FP32_MASTER", "") == "1":
        masters = [p.detach().float().clone().requires_grad_(False)
                   for p in trainable]
        print(f"[fp32-master] {sum(m.numel() for m in masters)/1e9:.2f}B "
              "master params", flush=True)
    opt_params = masters if masters is not None else trainable
    if NO_WD_EMB:                    # OLMo2: no weight decay on embed/head
        emb_ids = {id(p) for n, p in raw_model.named_parameters()
                   if "embed_tokens" in n or "lm_head" in n}
        flags = [id(p) in emb_ids for p in trainable]
        opt_params = [
            {"params": [q for q, fl in zip(opt_params, flags) if fl],
             "weight_decay": 0.0},
            {"params": [q for q, fl in zip(opt_params, flags) if not fl]}]
        print("[opt] weight decay off for embed/head", flush=True)
    ob = os.environ.get("OPT_8BIT", "")
    if ob in ("1", "paged"):
        import bitsandbytes as bnb
        # paged: states in CPU-pageable memory, migrated on demand — the
        # 32 GB-card option (5B fit probe OOM'd by 20 MiB on-GPU states)
        cls = bnb.optim.PagedAdamW8bit if ob == "paged" \
            else bnb.optim.AdamW8bit
        opt = cls(opt_params, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
        print(f"[opt] {cls.__name__} (bitsandbytes)", flush=True)
    else:
        opt = torch.optim.AdamW(opt_params, lr=LR, betas=(0.9, 0.95),
                                weight_decay=0.1, fused=True)
    steps = int(os.environ.get("STEPS", "3000"))
    bs = int(os.environ.get("BS", "8"))
    warm = WARMUP
    tps = world * bs * SEQ                  # tokens per optimizer step
    tok_state = {"seen": TOKENS_SEEN_INIT}

    def ckpt_state(step):
        """Versioned, uniform checkpoint schema — every save path uses this
        (review finding: ad-hoc dicts had drifted apart)."""
        return {"schema": 2,
                "model": raw_model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "tokens_seen": tok_state["seen"],
                "geometry": {"H": H, "NL": NL, "HEADS": HEADS,
                             "Q_LORA": Q_LORA, "MOE_INTER": MOE_INTER,
                             "SEQ": SEQ, "tps": tps},
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state(dev),
                        "numpy": np.random.get_state(),
                        "data_rng": rng.bit_generator.state}}
    global EOS_ID
    if tok.eos_token_id is not None:
        EOS_ID = tok.eos_token_id
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT / f"{SAVE_NAME}_ckpt.pt"
    import signal
    term = {"flag": False}
    spike_state = {"ema": None, "skips": 0}
    signal.signal(signal.SIGTERM, lambda *_: term.update(flag=True))
    start = 0
    if RESUME_PT:
        raw_model.load_state_dict(torch.load(RESUME_PT, map_location=dev,
                                             weights_only=False),
                              strict=True)
        print(f"[resume-weights] {RESUME_PT}", flush=True)
        if ckpt_path.exists():
            print(f"[resume] WARNING: stale {ckpt_path} exists and will "
                  "OVERRIDE RESUME_PT weights below — delete it if this "
                  "is a fresh stage", flush=True)
    if ckpt_path.exists():
        # weights_only=False: our own trusted file; the rng entry contains
        # numpy state objects the safe-loader rejects (torch>=2.6 default)
        state = torch.load(ckpt_path, map_location=dev, weights_only=False)
        raw_model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["step"] + 1
        if "tokens_seen" in state and not TOKENS_SEEN_INIT:
            tok_state["seen"] = state["tokens_seen"]
        if "rng" in state:            # ~10KB insurance (HF-Trainer practice)
            try:
                torch.set_rng_state(state["rng"]["torch"])
                torch.cuda.set_rng_state(state["rng"]["cuda"], dev)
                np.random.set_state(state["rng"]["numpy"])
                if "data_rng" in state["rng"]:
                    rng.bit_generator.state = state["rng"]["data_rng"]
            except Exception as exc:
                print(f"[resume] rng restore skipped: {exc}", flush=True)
        print(f"[resume] from step {start}", flush=True)
    if masters is not None:
        # masters were cloned from INIT weights before any resume load;
        # refresh or the first opt.step rolls weights back to random
        with torch.no_grad():
            for m, p in zip(masters, trainable):
                m.copy_(p.detach().float())
    if is_main:
        # incarnation banner: one ledger line per trainer start identifying
        # node + shape — the additive provenance record across preemptions
        # and tier changes (plotted as vertical markers)
        import socket
        from datetime import datetime, timezone
        gpu = torch.cuda.get_device_name(0).replace(" ", "_")
        print(f"[incarnation] step={start} tokens={tok_state['seen']} "
              f"host={socket.gethostname()} gpu={gpu}x{world} bs={bs} "
              f"seq={SEQ} tps={tps} "
              f"time={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
              flush=True)
    if TOKEN_BUDGET:
        steps = start + max(0, math.ceil(
            (TOKEN_BUDGET - tok_state["seen"]) / tps))
        print(f"[tokens] {tok_state['seen']/1e9:.3f}B seen of "
              f"{TOKEN_BUDGET/1e9:.3f}B; {tps} tok/step -> "
              f"end at step {steps}", flush=True)
    for step in range(start, steps):
        # schedule position: token-fraction when TOKEN_BUDGET set (tier-
        # agnostic resume), else step-fraction (legacy)
        if TOKEN_BUDGET:
            frac = min(1.0, tok_state["seen"] / TOKEN_BUDGET)
            wfrac = min(1.0, (tok_state["seen"] + tps) / (warm * tps))
        else:
            frac = step / steps
            wfrac = (step + 1) / warm
        if LR_SCHED == "wsd":
            f = 1.0 if frac < (1 - WSD_DECAY_FRAC) else max(
                0.1, 1.0 - 0.9 * (frac - (1 - WSD_DECAY_FRAC))
                / max(1e-9, WSD_DECAY_FRAC))
        else:
            f = 0.5 * (1 + math.cos(math.pi * frac))
        lr = LR * min(1.0, wfrac) * f
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)   # free last step's grads BEFORE
        bmask = None
        if mix is not None:                   # building this step's graph
            step_rng = np.random.default_rng(
                (20260805, rank, step)) if DET_DATA else rng
            batch, bmask = mix.batch(bs, SEQ, step_rng)
            batch = batch.to(dev)
            bmask = bmask.to(dev) if bmask is not None else None
        else:
            i = torch.randint(0, len(data) - SEQ - 2, (bs,))
            batch = torch.stack([data[j:j + SEQ + 2] for j in i]).to(dev)
        tok_state["seen"] += tps
        ids, tgt = batch[:, :SEQ], batch[:, 1:SEQ + 1]
        mtp_tgt = batch[:, 2:SEQ + 2][:, :SEQ - 1]
        if bmask is not None:                 # SFT: loss on assistant tokens
            tgt = tgt.masked_fill(bmask[:, 1:SEQ + 1] == 0, -100)
            mtp_tgt = mtp_tgt.masked_fill(
                bmask[:, 2:SEQ + 2][:, :SEQ - 1] == 0, -100)
        qn_params = None
        if QNOISE > 0:
            # seeded regeneration: same noise is re-created to subtract
            # after backward — zero storage for 4B+ params
            qn_params = []
            with torch.no_grad():
                for i, p in enumerate(raw_model.parameters()):
                    if not (p.requires_grad and p.dim() >= 2
                            and p.numel() >= 512 * 512):
                        continue
                    s = float(QNOISE * p.std())
                    qn_params.append((i, p, s))
                    g = torch.Generator(device=p.device)
                    g.manual_seed(step * 10007 + i)
                    p.add_(torch.randn(p.shape, generator=g, device=p.device,
                                       dtype=p.dtype) * s)
        h, mtp_h = model(ids)
        loss_main = ce_chunked(raw_model.lm_head, h, tgt)
        loss_mtp = ce_chunked(raw_model.lm_head, mtp_h, mtp_tgt)
        aux = sum(b.mlp.aux for b in raw_model.layers if b.is_moe)
        aux = aux + raw_model.mtp_block.mlp.aux
        mtp_w = MTP_W if MTP_W_END <= 0 else \
            MTP_W + (MTP_W_END - MTP_W) * (frac if TOKEN_BUDGET
                                           else step / max(1, steps - 1))
        loss = loss_main + mtp_w * loss_mtp + AUX_COEF * aux
        zc = float(os.environ.get("ZLOSS", "0"))
        if zc > 0:
            zs = [b.mlp.zloss for b in raw_model.layers
                  if b.is_moe and getattr(b.mlp, "zloss", None) is not None]
            if getattr(raw_model.mtp_block.mlp, "zloss", None) is not None:
                zs.append(raw_model.mtp_block.mlp.zloss)
            if zs:
                loss = loss + zc * (sum(zs) / len(zs))
        if DISTILL:
            dl = [b.self_attn.distill_loss for b in raw_model.layers
                  if b.self_attn.distill_loss is not None]
            if raw_model.mtp_block.self_attn.distill_loss is not None:
                dl.append(raw_model.mtp_block.self_attn.distill_loss)
            if dl:
                distill = sum(dl) / len(dl)
                loss = loss + 0.1 * distill
                if step % 50 == 0:
                    print(f"    [distill] kl {distill.item():.4f}",
                          flush=True)
        loss.backward()
        if qn_params is not None:
            # regenerate identical noise and remove it: grads were computed
            # AT the perturbed point, but the update applies to clean weights
            with torch.no_grad():
                for i, p, s in qn_params:
                    g = torch.Generator(device=p.device)
                    g.manual_seed(step * 10007 + i)
                    p.sub_(torch.randn(p.shape, generator=g, device=p.device,
                                       dtype=p.dtype) * s)
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0))
        if os.environ.get("SKIP_SPIKES", "") == "1" and step > 100:
            ema = spike_state["ema"]
            if ema is not None and gnorm > 4.0 * ema:
                spike_state["skips"] += 1
                print(f"[spike] step {step}: gnorm {gnorm:.2f} > 4x ema "
                      f"{ema:.2f} — update SKIPPED "
                      f"(total {spike_state['skips']})", flush=True)
                if term["flag"]:          # don't let a spike streak defer
                    break                 # the SIGTERM snapshot forever
                continue
            spike_state["ema"] = gnorm if ema is None \
                else 0.99 * ema + 0.01 * gnorm
        if masters is not None:
            # bf16 grads -> fp32 masters; step in fp32; copy back to bf16.
            # Kills the bf16 dead-zone (updates < ~0.2% are otherwise lost).
            with torch.no_grad():
                for m, p in zip(masters, trainable):
                    if p.grad is None:   # e.g. params outside this stage's
                        continue         # loss graph (mtp indexer w/o distill)
                    m.grad = p.grad.float() if m.grad is None \
                        else m.grad.copy_(p.grad.float())
                opt.step()
                for m, p in zip(masters, trainable):
                    p.copy_(m.to(p.dtype))
        else:
            opt.step()
        if BIAS_BALANCE > 0:
            # DSv3 aux-loss-free balancing: nudge the (frozen, exported)
            # e_score_correction_bias toward uniform expert load. Done here
            # (not in forward) so grad-ckpt recompute can't double-apply.
            with torch.no_grad():
                moes = [b.mlp for b in raw_model.layers if b.is_moe]
                moes.append(raw_model.mtp_block.mlp)
                for m in moes:
                    c = getattr(m, "last_counts", None)
                    if c is None:
                        continue
                    c = c.float()
                    if ddp:
                        torch.distributed.all_reduce(c)
                    m.e_score_correction_bias.add_(
                        BIAS_BALANCE * torch.sign(c.mean() - c))
        if term["flag"]:
            if is_main:
                torch.save(ckpt_state(step),
                           str(ckpt_path) + ".tmp")
                os.replace(str(ckpt_path) + ".tmp", ckpt_path)
                push_ckpt(step)
                print(f"[sigterm] snapshot at step {step}, exiting",
                      flush=True)
            break
        if (step % 50 == 0 or step == steps - 1) and is_main:
            print(f"[{step}/{steps}] loss {loss_main.item():.3f} "
                  f"mtp {loss_mtp.item():.3f} aux {aux.item():.2f} "
                  f"gnorm {gnorm:.2f} "
                  f"lr {lr:.1e} {(time.time()-t0)/(step-start+1):.2f}s/step",
                  flush=True)
        if step % 250 == 249 and is_main:
            # SIGSEGV insurance; tmp+replace = atomic vs crash AND vs the
            # in-flight push thread (which reads a hardlink of the old inode)
            torch.save({"model": raw_model.state_dict(),
                        "opt": opt.state_dict(), "step": step,
                        "rng": {"torch": torch.get_rng_state(),
                                "cuda": torch.cuda.get_rng_state(dev),
                                "numpy": np.random.get_state()}},
                       str(ckpt_path) + ".tmp")
            os.replace(str(ckpt_path) + ".tmp", ckpt_path)
        if step % push_every == push_every - 1 and is_main:
            saver_busy = False
            if snap_state.get("saver_pid"):
                try:
                    pid, _ = os.waitpid(snap_state["saver_pid"], os.WNOHANG)
                    if pid == 0:
                        saver_busy = True
                    else:
                        snap_state["saver_pid"] = None
                except ChildProcessError:
                    snap_state["saver_pid"] = None
            if snap_state["saver"] and snap_state["saver"].is_alive():
                saver_busy = True
            if os.environ.get("SNAPSHOT_SAVE", "") == "1" and not saver_busy:
                ts = time.time()
                state = ckpt_state(step)
                snap_state["mirror"] = _snap_tree(state,
                                                  snap_state["mirror"])
                torch.cuda.synchronize()
                snap = snap_state["mirror"]
                print(f"[snap] paused {time.time()-ts:.2f}s", flush=True)

                def save_and_push(sn=snap, st=step):
                    tmp = str(ckpt_path) + ".tmpsnap"
                    torch.save(sn, tmp)         # mirror is free after this
                    os.replace(tmp, ckpt_path)  # atomic vs in-flight upload
                    push_ckpt(st)               # upload reads the file
                if os.environ.get("SNAPSHOT_FORK", "") == "1":
                    # process-based writer: a torch.save in a THREAD taxes
                    # every training step via the GIL during serialization;
                    # the forked child touches only CPU/pinned memory+files
                    # and does the upload itself (parent pushing here would
                    # race the child's write and upload a stale file)
                    pid = os.fork()
                    if pid == 0:
                        try:
                            # unique tmp: must NOT collide with the periodic
                            # save's .tmp (measured rename race)
                            tmp = f"{ckpt_path}.tmp{os.getpid()}"
                            torch.save(snap, tmp)
                            os.replace(tmp, ckpt_path)
                            if push_repo:
                                from huggingface_hub import HfApi
                                HfApi(token=os.environ["HF_TOKEN"]).\
                                    upload_file(
                                        path_or_fileobj=str(ckpt_path),
                                        path_in_repo=(f"checkpoints/"
                                                      f"{SAVE_NAME}_ckpt.pt"),
                                        repo_id=push_repo,
                                        commit_message=f"step {step}")
                                print(f"[push] step {step} -> {push_repo}",
                                      flush=True)
                        except Exception as exc:
                            print(f"[snap-fork] failed: {exc}", flush=True)
                        finally:
                            os._exit(0)
                    snap_state["saver_pid"] = pid
                else:
                    snap_state["saver"] = threading.Thread(
                        target=save_and_push, daemon=True)
                    snap_state["saver"].start()
            else:
                torch.save(ckpt_state(step),
                           str(ckpt_path) + ".tmp")
                os.replace(str(ckpt_path) + ".tmp", ckpt_path)
                push_ckpt(step)
        if step % eval_every == eval_every - 1 and is_main:
            run_val(step)
    if is_main:
        torch.save(raw_model.state_dict(), OUT / f"{SAVE_NAME}.pt")
    if ddp:
        torch.distributed.destroy_process_group()
    if push_state["t"] and push_state["t"].is_alive():
        print("[push] waiting for in-flight upload...", flush=True)
        push_state["t"].join()
    # quick greedy sample
    model.eval()
    with torch.no_grad():
        for prompt in ("Once upon a time", "MIT License\n\nCopyright"):
            ids = torch.tensor([tok.encode(prompt)], device=dev)
            for _ in range(30):
                h, _ = model(ids[:, -SEQ:])
                nxt = raw_model.lm_head(h[:, -1:]).argmax(-1)
                ids = torch.cat([ids, nxt], dim=1)
            print(f"[sample] {prompt!r} -> "
                  f"{tok.decode(ids[0])[len(prompt):][:90]!r}", flush=True)
    print(f"[mem] peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB "
          f"(reserved {torch.cuda.max_memory_reserved()/2**30:.1f})",
          flush=True)
    print("TRAIN-DONE", flush=True)


if __name__ == "__main__":
    main()
