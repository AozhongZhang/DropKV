"""Triton-accelerated DropKV prefill eviction scoring.

Three-kernel architecture (plus shared helper):
  Kernel A — FlashAttention-style forward: computes Y=PV and lse (online softmax).
             Single-CTA or split-K grid depending on GPU utilization.
  Kernel B — Score computation: recomputes logits from lse, accumulates per-token scores.
  Kernel C — Reduction: merges split-K partial online-softmax state (only when num_splits > 1).
  Shared   — _compute_logits_block: tiled Q@K^T called by Kernels A and B.

Supports Q[B,Hq,Lq,D] with K/V[B,Hkv,Lk,D] where Lq <= Lk (chunked prefill).
Reference function (pure PyTorch, no repeat_kv) included for correctness validation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Step 1: Pure-PyTorch reference (no repeat_kv)
# ---------------------------------------------------------------------------


def dropkv_scores_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int = 8,
    eps: float = 1e-6,
    match_pytorch_precision: bool = False,
) -> torch.Tensor:
    """Compute DropKV eviction scores without repeat_kv (mathematically exact).

    Uses contiguous GQA grouping: Q[:, hkv*g:(hkv+1)*g, -W:, :].
    All arithmetic in fp32 by default for maximum accuracy.

    Args:
        Q: [B, Hq, Lq, D]  K: [B, Hkv, Lk, D]  V: [B, Hkv, Lk, D]
            Lq may differ from Lk (chunked prefill). Only last W positions of Q
            are used.  Scores are indexed by key position (length Lk).
            Q must be tail-aligned with K/V: the last query position corresponds
            to the last key position (position Lk-1). Requires Lq >= W and Lk >= W.
        window_size: number of recent query positions used for scoring
        eps: numerical stability constant for W computation
        match_pytorch_precision: if True, cast softmax output to input dtype
            before W computation (matches deployed PyTorch path behavior).
            Default False — reference computes mathematically exact scores.

    Returns:
        scores: [B, Hkv, Lk] fp32
    """
    B, Hq, Lq, D = Q.shape
    Hkv = K.shape[1]
    Lk = K.shape[2]
    g = Hq // Hkv
    W = window_size
    assert Lq >= W, f"Lq ({Lq}) must be >= window_size ({W})"
    assert Lk >= W, f"Lk ({Lk}) must be >= window_size ({W})"
    assert Lq <= Lk, f"Lq ({Lq}) must be <= Lk ({Lk}) (Q must be tail-aligned suffix)"

    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()

    scores = torch.zeros(B, Hkv, Lk, device=Q.device, dtype=torch.float32)

    for hkv in range(Hkv):
        Q_group = Q_f[:, hkv * g : (hkv + 1) * g, -W:, :]  # [B, g, W, D]
        K_h = K_f[:, hkv : hkv + 1, :, :]  # [B, 1, Lk, D]
        V_h = V_f[:, hkv : hkv + 1, :, :]  # [B, 1, Lk, D]

        for gi in range(g):
            Q_w = Q_group[:, gi : gi + 1, :, :]  # [B, 1, W, D]
            attn_logits = (Q_w @ K_h.transpose(-1, -2)) / math.sqrt(D)

            mask = torch.full(
                (W, Lk), float("-inf"), device=Q.device, dtype=torch.float32
            )
            mask = torch.triu(mask, diagonal=Lk - W + 1)
            attn_logits = attn_logits + mask

            p = F.softmax(attn_logits, dim=-1, dtype=torch.float32)
            if match_pytorch_precision:
                # Round p to input dtype (matching PyTorch's softmax→bf16 cast),
                # then compute W in fp32. The bf16 rounding of p near 1.0 is the
                # critical step that determines attention-sink token rankings.
                p = p.to(Q.dtype).float()
            y = p @ V_h
            W_mat = (p / (1.0 - p + eps)) ** 2

            W_sum = W_mat.sum(dim=-2)
            v_norm2 = (V_h ** 2).sum(dim=-1)
            termA = W_sum * v_norm2

            WT_y = W_mat.transpose(-1, -2) @ y
            termB = 2.0 * (V_h * WT_y).sum(dim=-1)

            y_norm2 = (y ** 2).sum(dim=-1)
            termC = (W_mat.transpose(-1, -2) @ y_norm2.unsqueeze(-1)).squeeze(-1)

            scores[:, hkv : hkv + 1, :] += termA - termB + termC

    scores /= g
    return scores


# ---------------------------------------------------------------------------
# Step 2: Shared Triton logit helper + Kernel A
# ---------------------------------------------------------------------------


@triton.jit
def _compute_logits_block(
    q_base,
    K_ptr,
    offs_n,
    mask_m,
    mask_n,
    causal_bound,
    inv_sqrt_d,
    stride_qd,
    stride_ks,
    stride_kd,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    EVICT_K: tl.constexpr = 0,
):
    """Shared logit computation — called by BOTH Kernel A and Kernel B.

    Computes: logits = (Q_tile @ K_block^T) * inv_sqrt_d
    with causal + invalid-row masking applied.

    Args:
        q_base: [BLOCK_M] scattered Q row base pointers.
        K_ptr: base pointer to K[b, hkv, 0, 0].
        offs_n: [BLOCK_N] key position offsets.
        mask_m: [BLOCK_M] valid row mask.
        mask_n: [BLOCK_N] valid column mask (j < L).
        causal_bound: [BLOCK_M] per-row causal upper bound (inclusive).
        inv_sqrt_d: 1/sqrt(D).
        stride_qd: Q head_dim stride.
        stride_ks, stride_kd: K seq and head_dim strides.
        EVICT_K: 0 = evict_first (Kernel A streaming), 1 = evict_last (Kernel B L1 reuse).

    Returns:
        logits: [BLOCK_M, BLOCK_N] fp32 with causal + invalid-row masking.
    """
    dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D

        # Keep native dtype (bf16/fp16) for tensor-core dot with fp32 accumulation.
        # Same pattern as FlashAttention — avoids fp32 tile buffers that inflate
        # register pressure in both Kernel A and B.
        q_chunk = tl.load(
            q_base[:, None] + offs_k[None, :] * stride_qd,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        if EVICT_K == 0:
            k_chunk = tl.load(
                K_ptr + offs_n[None, :] * stride_ks + offs_k[:, None] * stride_kd,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            k_chunk = tl.load(
                K_ptr + offs_n[None, :] * stride_ks + offs_k[:, None] * stride_kd,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )

        dot += tl.dot(q_chunk, k_chunk, allow_tf32=ALLOW_TF32)

    logits = dot * inv_sqrt_d
    logits = tl.where(offs_n[None, :] <= causal_bound[:, None], logits, float("-inf"))
    logits = tl.where(mask_m[:, None], logits, float("-inf"))
    return logits


@triton.jit
def _dropkv_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Y_ptr, lse_ptr,
    Hq, Hkv, L, M, g, W,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_yb, stride_yh, stride_ym, stride_yd,
    stride_lb, stride_lh, stride_lm,
    inv_sqrt_d,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Kernel A: FlashAttention-style forward pass.

    Grid: (B * Hkv,) — one program per (batch, kv_head).
    Streams over all L keys, accumulating Y=PV and lse via online softmax.

    Outputs:
        Y[B, Hkv, M, D] in input dtype — attention output per M query row.
        lse[B, Hkv, M] fp32 — log-sum-exp for softmax reconstruction.
    """
    pid = tl.program_id(0)
    b_idx = pid // Hkv
    hkv_idx = pid % Hkv

    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Scattered Q addressing with clamped head indices
    # Q_ptr points to Q_window[B, Hq, W, D] (pre-sliced last W positions)
    q_head = tl.minimum(hkv_idx * g + offs_m // W, Hq - 1)
    q_pos = offs_m % W  # index into Q_window (0..W-1)
    causal_bound = L - W + q_pos  # causal bound uses original L

    q_base = Q_ptr + b_idx * stride_qb + q_head * stride_qh + q_pos * stride_qs
    k_base = K_ptr + b_idx * stride_kb + hkv_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + hkv_idx * stride_vh

    # NaN-safe online softmax init
    m_i = tl.where(mask_m, float("-inf"), 0.0)
    s_i = tl.where(mask_m, 0.0, 1.0)
    acc_y = tl.zeros([BLOCK_M, D], tl.float32)

    offs_d = tl.arange(0, D)

    for j0 in range(0, L, BLOCK_N):
        j0 = tl.multiple_of(j0, BLOCK_N)
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L

        logits = _compute_logits_block(
            q_base, k_base, offs_n, mask_m, mask_n, causal_bound, inv_sqrt_d,
            stride_qd, stride_ks, stride_kd,
            D, BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        # Online softmax
        block_max = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - new_m)
        p = tl.exp(logits - new_m[:, None])

        s_i = s_i * alpha + tl.sum(p, axis=1)
        acc_y = acc_y * alpha[:, None]

        # p @ V_tile: [BLOCK_M, BLOCK_N] @ [BLOCK_N, D] -> [BLOCK_M, D]
        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        ).to(tl.float32)
        acc_y += tl.dot(p.to(tl.float32), v_tile, allow_tf32=ALLOW_TF32)

        m_i = new_m

    # Finalize
    s_safe = tl.maximum(s_i, 1e-40)
    lse = m_i + tl.log(s_safe)
    y_out = acc_y / s_i[:, None]

    # Store lse [BLOCK_M] fp32
    lse_base = lse_ptr + b_idx * stride_lb + hkv_idx * stride_lh
    tl.store(lse_base + offs_m * stride_lm, lse, mask=mask_m)

    # Store Y [BLOCK_M, D] in input dtype (auto-cast via pointer type)
    y_base = Y_ptr + b_idx * stride_yb + hkv_idx * stride_yh
    tl.store(
        y_base + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd,
        y_out.to(Y_ptr.type.element_ty),
        mask=mask_m[:, None],
    )


@triton.jit
def _dropkv_attn_fwd_split_kernel(
    Q_ptr, K_ptr, V_ptr, m_partial_ptr, s_partial_ptr, y_partial_ptr,
    Hq, Hkv, L, M, g, W,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_mb, stride_mh, stride_ms, stride_mm,
    stride_sb_p, stride_sh_p, stride_ss, stride_sm_p,
    stride_yb_p, stride_yh_p, stride_ys, stride_ym_p, stride_yd_p,
    inv_sqrt_d,
    tiles_per_split,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Split-K Kernel A: each program handles a contiguous range of key tiles.

    Grid: (B * Hkv, num_splits).
    Outputs partial online-softmax state (m, s, acc_y) in fp32.
    """
    pid_head = tl.program_id(0)
    pid_split = tl.program_id(1)
    b_idx = pid_head // Hkv
    hkv_idx = pid_head % Hkv

    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Scattered Q addressing (identical to single-CTA Kernel A)
    q_head = tl.minimum(hkv_idx * g + offs_m // W, Hq - 1)
    q_pos = offs_m % W
    causal_bound = L - W + q_pos

    q_base = Q_ptr + b_idx * stride_qb + q_head * stride_qh + q_pos * stride_qs
    k_base = K_ptr + b_idx * stride_kb + hkv_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + hkv_idx * stride_vh

    # NaN-safe online softmax init (identical to single-CTA)
    m_i = tl.where(mask_m, float("-inf"), 0.0)
    s_i = tl.where(mask_m, 0.0, 1.0)
    acc_y = tl.zeros([BLOCK_M, D], tl.float32)

    offs_d = tl.arange(0, D)

    # Contiguous key range for this split
    split_start = pid_split * tiles_per_split * BLOCK_N
    split_end = tl.minimum(split_start + tiles_per_split * BLOCK_N, L)

    for j0 in range(split_start, split_end, BLOCK_N):
        j0 = tl.multiple_of(j0, BLOCK_N)
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L

        logits = _compute_logits_block(
            q_base, k_base, offs_n, mask_m, mask_n, causal_bound, inv_sqrt_d,
            stride_qd, stride_ks, stride_kd,
            D, BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        # Online softmax — NaN-safe for fully-masked blocks.
        # When window_size > BLOCK_N, a split can have ALL keys beyond the
        # causal bound for some rows, giving block_max = -inf.  If m_i is
        # also -inf (first block), exp(-inf - (-inf)) = NaN.  Guard: when
        # new_m is -inf, set alpha=1 (keep old state) and p=0 (no contrib).
        block_max = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, block_max)
        all_masked = (new_m == float('-inf'))
        alpha = tl.exp(tl.where(all_masked, 0.0, m_i - new_m))
        p = tl.exp(tl.where(all_masked[:, None], float('-inf'), logits - new_m[:, None]))

        s_i = s_i * alpha + tl.sum(p, axis=1)
        acc_y = acc_y * alpha[:, None]

        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        ).to(tl.float32)
        acc_y += tl.dot(p.to(tl.float32), v_tile, allow_tf32=ALLOW_TF32)

        m_i = new_m

    # Store partial state (all fp32) — no normalization, no dtype cast
    partial_base_m = m_partial_ptr + b_idx * stride_mb + hkv_idx * stride_mh + pid_split * stride_ms
    tl.store(partial_base_m + offs_m * stride_mm, m_i, mask=mask_m)

    partial_base_s = s_partial_ptr + b_idx * stride_sb_p + hkv_idx * stride_sh_p + pid_split * stride_ss
    tl.store(partial_base_s + offs_m * stride_sm_p, s_i, mask=mask_m)

    partial_base_y = y_partial_ptr + b_idx * stride_yb_p + hkv_idx * stride_yh_p + pid_split * stride_ys
    tl.store(
        partial_base_y + offs_m[:, None] * stride_ym_p + offs_d[None, :] * stride_yd_p,
        acc_y,
        mask=mask_m[:, None],
    )


@triton.jit
def _dropkv_reduce_kernel(
    m_partial_ptr, s_partial_ptr, y_partial_ptr,
    Y_ptr, lse_ptr,
    Hkv, M, num_splits,
    stride_mb, stride_mh, stride_ms, stride_mm,
    stride_sb_p, stride_sh_p, stride_ss, stride_sm_p,
    stride_yb_p, stride_yh_p, stride_ys, stride_ym_p, stride_yd_p,
    stride_yb, stride_yh, stride_ym, stride_yd,
    stride_lb, stride_lh, stride_lm,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Kernel C: merge split-K partials via online-softmax rescaling.

    Grid: (B * Hkv,) — one program per (batch, kv_head).
    Reads partial (m, s, acc_y) from each split, produces final Y and lse.
    """
    pid = tl.program_id(0)
    b_idx = pid // Hkv
    hkv_idx = pid % Hkv

    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_d = tl.arange(0, D)

    # Load split 0 as initial state
    base_m0 = m_partial_ptr + b_idx * stride_mb + hkv_idx * stride_mh + 0 * stride_ms
    base_s0 = s_partial_ptr + b_idx * stride_sb_p + hkv_idx * stride_sh_p + 0 * stride_ss
    base_y0 = y_partial_ptr + b_idx * stride_yb_p + hkv_idx * stride_yh_p + 0 * stride_ys

    m_acc = tl.load(base_m0 + offs_m * stride_mm, mask=mask_m, other=0.0)
    s_acc = tl.load(base_s0 + offs_m * stride_sm_p, mask=mask_m, other=1.0)
    y_acc = tl.load(
        base_y0 + offs_m[:, None] * stride_ym_p + offs_d[None, :] * stride_yd_p,
        mask=mask_m[:, None], other=0.0,
    )

    # Merge remaining splits
    for split_idx in range(1, num_splits):
        base_m = m_partial_ptr + b_idx * stride_mb + hkv_idx * stride_mh + split_idx * stride_ms
        base_s = s_partial_ptr + b_idx * stride_sb_p + hkv_idx * stride_sh_p + split_idx * stride_ss
        base_y = y_partial_ptr + b_idx * stride_yb_p + hkv_idx * stride_yh_p + split_idx * stride_ys

        m_part = tl.load(base_m + offs_m * stride_mm, mask=mask_m, other=0.0)
        s_part = tl.load(base_s + offs_m * stride_sm_p, mask=mask_m, other=1.0)
        y_part = tl.load(
            base_y + offs_m[:, None] * stride_ym_p + offs_d[None, :] * stride_yd_p,
            mask=mask_m[:, None], other=0.0,
        )

        new_m = tl.maximum(m_acc, m_part)
        alpha = tl.exp(m_acc - new_m)
        beta = tl.exp(m_part - new_m)

        s_acc = s_acc * alpha + s_part * beta
        y_acc = y_acc * alpha[:, None] + y_part * beta[:, None]
        m_acc = new_m

    # Finalize
    s_safe = tl.maximum(s_acc, 1e-40)
    lse = m_acc + tl.log(s_safe)
    y_out = y_acc / s_acc[:, None]

    # Store lse [BLOCK_M] fp32
    lse_base = lse_ptr + b_idx * stride_lb + hkv_idx * stride_lh
    tl.store(lse_base + offs_m * stride_lm, lse, mask=mask_m)

    # Store Y [BLOCK_M, D] in input dtype (auto-cast via pointer type)
    y_base = Y_ptr + b_idx * stride_yb + hkv_idx * stride_yh
    tl.store(
        y_base + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd,
        y_out.to(Y_ptr.type.element_ty),
        mask=mask_m[:, None],
    )


# ---------------------------------------------------------------------------
# Step 3: Kernel B — Score computation (2D grid)
# ---------------------------------------------------------------------------


@triton.jit
def _dropkv_score_kernel(
    Q_ptr, K_ptr, V_ptr, Y_ptr, lse_ptr, score_ptr,
    Hq, Hkv, L, M, g, W,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_yb, stride_yh, stride_ym, stride_yd,
    stride_lb, stride_lh, stride_lm,
    stride_sb, stride_sh, stride_sj,
    inv_sqrt_d,
    eps,
    D: tl.constexpr,
    BLOCK_M_TILE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr = 0,
):
    """Kernel B: Score computation with 2D grid.

    Grid: (B * Hkv, ceil_div(L, BLOCK_N)).
    Each program handles one (b, hkv, j_block) tile.

    For each j-block of BLOCK_N keys:
      - Tiles over M query rows in BLOCK_M_TILE chunks
      - Recomputes logits from lse to get p, then w = (p/(1-p+eps))^2
      - Computes dist2 = ||v_j - y_m||^2 via norms + dot
      - Accumulates score_j = sum_m(w_mj * dist2_mj)

    DTYPE_ID: 0=no round, 1=bf16 round-trip, 2=fp16 round-trip.
        When >0, p is cast to the input dtype before W computation,
        matching PyTorch's F.softmax(...).to(input_dtype) behavior.
    """
    pid_head = tl.program_id(0)
    pid_j = tl.program_id(1)
    b_idx = pid_head // Hkv
    hkv_idx = pid_head % Hkv

    j0 = pid_j * BLOCK_N
    offs_n = j0 + tl.arange(0, BLOCK_N)
    mask_n = offs_n < L

    # Base pointers for this (b, hkv)
    k_base = K_ptr + b_idx * stride_kb + hkv_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + hkv_idx * stride_vh
    y_base = Y_ptr + b_idx * stride_yb + hkv_idx * stride_yh
    lse_base = lse_ptr + b_idx * stride_lb + hkv_idx * stride_lh

    # Load V block once for this j-block — used for both v_norm2 and y@v^T
    offs_d = tl.arange(0, D)
    v_block_native = tl.load(
        v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None],
        other=0.0,
        eviction_policy="evict_last",
    )  # keep native dtype (bf16/fp16) for tensor-core dot
    v_block_f32 = v_block_native.to(tl.float32)
    v_norm2 = tl.sum(v_block_f32 * v_block_f32, axis=1)  # [BLOCK_N]

    score_acc = tl.zeros([BLOCK_N], tl.float32)

    # M-tile loop: iterate over M query rows in BLOCK_M_TILE chunks
    for m0 in range(0, M, BLOCK_M_TILE):
        offs_m_tile = m0 + tl.arange(0, BLOCK_M_TILE)
        mask_m_tile = offs_m_tile < M

        # Scattered Q addressing (same as Kernel A)
        # Q_ptr points to Q_window[B, Hq, W, D] (pre-sliced last W positions)
        q_head = tl.minimum(hkv_idx * g + offs_m_tile // W, Hq - 1)
        q_pos = offs_m_tile % W  # index into Q_window (0..W-1)
        causal_bound = L - W + q_pos  # causal bound uses original L

        q_base_tile = (
            Q_ptr + b_idx * stride_qb + q_head * stride_qh + q_pos * stride_qs
        )

        # Load lse for this m-tile [BLOCK_M_TILE]
        lse_tile = tl.load(
            lse_base + offs_m_tile * stride_lm,
            mask=mask_m_tile,
            other=0.0,
        )

        # Load Y tile [BLOCK_M_TILE, D] — keep native dtype for tensor-core Y@V^T
        y_tile_native = tl.load(
            y_base + offs_m_tile[:, None] * stride_ym + offs_d[None, :] * stride_yd,
            mask=mask_m_tile[:, None],
            other=0.0,
        )
        y_tile = y_tile_native.to(tl.float32)
        y_norm2_tile = tl.sum(y_tile * y_tile, axis=1)  # [BLOCK_M_TILE]

        # Compute logits via shared helper (identical dot path to Kernel A)
        logits = _compute_logits_block(
            q_base_tile, k_base, offs_n, mask_m_tile, mask_n,
            causal_bound, inv_sqrt_d,
            stride_qd, stride_ks, stride_kd,
            D, BLOCK_M_TILE, BLOCK_N, BLOCK_K, ALLOW_TF32,
            EVICT_K=1,  # evict_last for L1 reuse across m-tiles
        )

        # Reconstruct p from lse: p = exp(logits - lse)
        p = tl.exp(logits - lse_tile[:, None])
        p = tl.where(mask_m_tile[:, None], p, 0.0)
        # Clamp to [0, 1] — NOT [0, 1-eps]
        p = tl.minimum(tl.maximum(p, 0.0), 1.0)

        # Match PyTorch precision: PyTorch does F.softmax(...).to(input_dtype),
        # casting p to bf16/fp16 before W computation. The bf16 rounding of p
        # near 1.0 changes W by orders of magnitude at attention-sink tokens.
        # After rounding p, W is computed in fp32 for numerical stability.
        if DTYPE_ID == 1:
            p = p.to(tl.bfloat16).to(tl.float32)
        elif DTYPE_ID == 2:
            p = p.to(tl.float16).to(tl.float32)

        # w = (p / (1 - p + eps))^2  — fp32 throughout
        ratio = p / (1.0 - p + eps)
        w = ratio * ratio  # [BLOCK_M_TILE, BLOCK_N]

        # dist2 = ||v_j - y_m||^2 = ||v_j||^2 - 2*<y_m, v_j> + ||y_m||^2
        # y_dot_v: [BLOCK_M_TILE, BLOCK_N] via tensor-core dot (native dtype)
        # v_block_native loaded once before the m-tile loop (no reload)
        # y_tile @ v_block^T: [BLOCK_M_TILE, D] @ [D, BLOCK_N] -> [BLOCK_M_TILE, BLOCK_N]
        y_dot_v = tl.dot(y_tile_native, tl.trans(v_block_native), allow_tf32=ALLOW_TF32)

        dist2 = v_norm2[None, :] - 2.0 * y_dot_v + y_norm2_tile[:, None]
        dist2 = tl.maximum(dist2, 0.0)  # clamp numerical noise from mixed-precision

        # Accumulate: score_j += sum_m(w_mj * dist2_mj)
        score_acc += tl.sum(w * dist2, axis=0)

    # Average over g query groups
    score_out = score_acc / g

    # Store scores [BLOCK_N]
    score_base = score_ptr + b_idx * stride_sb + hkv_idx * stride_sh
    tl.store(score_base + offs_n * stride_sj, score_out, mask=mask_n)


# ---------------------------------------------------------------------------
# Step 4: Python wrapper
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    """Round up to next power of 2 (min 16 for tl.dot)."""
    n = max(n, 16)
    return 1 << (n - 1).bit_length()


def _compute_num_splits(B: int, Hkv: int, L: int, BLOCK_N: int, device: torch.device) -> int:
    """Compute optimal num_splits for split-K Kernel A.

    Targets 2x SM count for good occupancy. Caps by minimum tiles per split
    to avoid oversplitting at small L. Returns 1 when splitting isn't beneficial.
    """
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    total_tiles = (L + BLOCK_N - 1) // BLOCK_N
    target_ctas = 2 * num_sms
    base_ctas = B * Hkv
    if base_ctas >= target_ctas:
        return 1
    wanted = math.ceil(target_ctas / base_ctas)
    min_tiles_per_split = 4
    max_by_work = max(1, total_tiles // min_tiles_per_split)
    num_splits = min(wanted, max_by_work, 64)
    return max(1, num_splits)


def dropkv_scores_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int = 8,
    eps: float = 1e-6,
    allow_tf32: bool = True,
    match_pytorch_precision: bool = True,
) -> torch.Tensor:
    """Compute DropKV eviction scores using Triton kernels.

    Args:
        Q: [B, Hq, Lq, D] query states (bf16 or fp16, need not be contiguous —
            only the last ``window_size`` positions are used and sliced internally).
            Lq may differ from Lk (e.g. chunked prefill where Q is one chunk
            but K/V span the full accumulated sequence).  Q must be tail-aligned
            with K/V: the last query position corresponds to key position Lk-1.
            Requires Lq >= window_size.
        K: [B, Hkv, Lk, D] key states (same dtype as Q, contiguous). Requires Lk >= window_size.
        V: [B, Hkv, Lk, D] value states (same dtype as Q, contiguous)
        window_size: number of recent query positions for scoring
        eps: stability constant for w computation
        allow_tf32: allow TF32 tensor cores for fp32 matmuls
        match_pytorch_precision: if True (default), cast p to input dtype
            before W computation to match the deployed PyTorch path's
            F.softmax(...).to(dtype) behavior. Required for topk parity
            with the PyTorch scoring path. Set False for mathematically
            exact fp32 scores (useful for research/analysis).

    Returns:
        scores: [B, Hkv, Lk] fp32
    """
    B, Hq, Lq, D = Q.shape
    Hkv = K.shape[1]
    Lk = K.shape[2]

    # --- Guards ---
    assert isinstance(window_size, int) and window_size > 0, (
        f"window_size must be a positive int, got {window_size}"
    )
    assert Lq >= window_size, (
        f"Lq ({Lq}) must be >= window_size ({window_size})"
    )
    assert Lk >= window_size, (
        f"Lk ({Lk}) must be >= window_size ({window_size})"
    )
    assert Lq <= Lk, (
        f"Lq ({Lq}) must be <= Lk ({Lk}) (Q must be tail-aligned suffix)"
    )
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"
    assert Q.is_cuda, "Inputs must be on CUDA"
    assert Q.device == K.device == V.device, "All inputs must be on same device"
    assert K.is_contiguous() and V.is_contiguous(), (
        "K and V must be contiguous (Q is sliced internally)"
    )
    assert Q.dtype == K.dtype == V.dtype, "All inputs must have same dtype"
    assert Q.dtype in (torch.float16, torch.bfloat16), (
        f"Input dtype must be fp16 or bf16, got {Q.dtype}"
    )
    assert K.shape == V.shape, (
        f"K and V shape mismatch: K={K.shape}, V={V.shape}"
    )
    assert K.shape[0] == B and K.shape[3] == D, (
        f"K batch/head_dim mismatch: expected B={B}, D={D}, got K={K.shape}"
    )

    g = Hq // Hkv
    W = window_size
    M = g * W  # total query rows per KV head

    # Block sizes
    BLOCK_K = 32
    assert D >= BLOCK_K, f"D ({D}) must be >= BLOCK_K ({BLOCK_K})"
    assert D & (D - 1) == 0, f"D ({D}) must be a power of 2 for Triton tl.arange"
    BLOCK_M = _next_power_of_2(M)
    BLOCK_N = 64

    # Kernel B: BLOCK_M_TILE=min(32, BLOCK_M) eliminates inner m-loop for Llama (M=32)
    BLOCK_M_TILE = min(32, BLOCK_M)

    inv_sqrt_d = 1.0 / math.sqrt(D)

    # DTYPE_ID for p round-trip: 0=no round, 1=bf16, 2=fp16
    if match_pytorch_precision:
        DTYPE_ID = 1 if Q.dtype == torch.bfloat16 else 2
    else:
        DTYPE_ID = 0

    # Pre-slice Q to last W positions — avoids copying the full Q tensor.
    # At L=65K Llama: 537 MB (full Q) → 0.066 MB (Q_window).
    Q_w = Q[:, :, -W:, :].contiguous()  # [B, Hq, W, D]

    # Allocate intermediates
    Y = torch.zeros(B, Hkv, M, D, device=Q.device, dtype=Q.dtype)
    lse = torch.zeros(B, Hkv, M, device=Q.device, dtype=torch.float32)
    scores = torch.zeros(B, Hkv, Lk, device=Q.device, dtype=torch.float32)

    # --- Launch Kernel A (single-CTA or split-K) ---
    num_splits = _compute_num_splits(B, Hkv, Lk, BLOCK_N, Q.device)

    if num_splits <= 1:
        # Fast path: existing single-CTA Kernel A (bit-identical to v1)
        grid_a = (B * Hkv,)
        _dropkv_attn_fwd_kernel[grid_a](
            Q_w, K, V, Y, lse,
            Hq, Hkv, Lk, M, g, W,
            Q_w.stride(0), Q_w.stride(1), Q_w.stride(2), Q_w.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            inv_sqrt_d,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            ALLOW_TF32=allow_tf32,
        )
    else:
        # Split-K path: parallelize Kernel A across key dimension
        total_tiles = triton.cdiv(Lk, BLOCK_N)
        tiles_per_split = triton.cdiv(total_tiles, num_splits)

        # Allocate fp32 partial buffers
        m_partial = torch.empty(B, Hkv, num_splits, BLOCK_M, device=Q.device, dtype=torch.float32)
        s_partial = torch.empty(B, Hkv, num_splits, BLOCK_M, device=Q.device, dtype=torch.float32)
        y_partial = torch.zeros(B, Hkv, num_splits, BLOCK_M, D, device=Q.device, dtype=torch.float32)

        # Launch split Kernel A
        grid_a = (B * Hkv, num_splits)
        _dropkv_attn_fwd_split_kernel[grid_a](
            Q_w, K, V, m_partial, s_partial, y_partial,
            Hq, Hkv, Lk, M, g, W,
            Q_w.stride(0), Q_w.stride(1), Q_w.stride(2), Q_w.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            m_partial.stride(0), m_partial.stride(1), m_partial.stride(2), m_partial.stride(3),
            s_partial.stride(0), s_partial.stride(1), s_partial.stride(2), s_partial.stride(3),
            y_partial.stride(0), y_partial.stride(1), y_partial.stride(2), y_partial.stride(3), y_partial.stride(4),
            inv_sqrt_d,
            tiles_per_split,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            ALLOW_TF32=allow_tf32,
        )

        # Launch Kernel C (reduction)
        grid_c = (B * Hkv,)
        _dropkv_reduce_kernel[grid_c](
            m_partial, s_partial, y_partial, Y, lse,
            Hkv, M, num_splits,
            m_partial.stride(0), m_partial.stride(1), m_partial.stride(2), m_partial.stride(3),
            s_partial.stride(0), s_partial.stride(1), s_partial.stride(2), s_partial.stride(3),
            y_partial.stride(0), y_partial.stride(1), y_partial.stride(2), y_partial.stride(3), y_partial.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D=D,
            BLOCK_M=BLOCK_M,
        )

        del m_partial, s_partial, y_partial

    # --- Launch Kernel B (heuristic config: BLOCK_M_TILE=min(32,M), BLOCK_N=64) ---
    grid_b = (B * Hkv, triton.cdiv(Lk, BLOCK_N))
    _dropkv_score_kernel[grid_b](
        Q_w, K, V, Y, lse, scores,
        Hq, Hkv, Lk, M, g, W,
        Q_w.stride(0), Q_w.stride(1), Q_w.stride(2), Q_w.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        scores.stride(0), scores.stride(1), scores.stride(2),
        inv_sqrt_d,
        eps,
        D=D,
        BLOCK_M_TILE=BLOCK_M_TILE,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ALLOW_TF32=allow_tf32,
        DTYPE_ID=DTYPE_ID,
    )

    return scores
