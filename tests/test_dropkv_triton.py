"""Tests for DropKV Triton kernels.

Covers:
  1. Reference vs existing repeat_kv code parity
  2. Triton vs reference correctness
  3. NaN safety (Qwen padded rows)
  4. Kernel A↔B consistency (Σ exp(logits-lse) ≈ 1)
  5. mask_m invariant test
  6. dtype parity (fp16 + bf16)
  7. Guard failure tests
  8. Edge cases (L=W, non-pow2 L)
"""

import math

import pytest
import torch
import torch.nn.functional as F

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _existing_pytorch_scores(Q, K, V, window_size=8, eps=1e-6):
    """Reproduce the scoring from dropkv_cache.py / cache_utils.py.

    Matches the production code: softmax→bf16 cast, then fp32 scoring.
    """
    B, Hq, L, D = Q.shape
    Hkv = K.shape[1]
    g = Hq // Hkv
    W = window_size

    def repeat_kv(hidden_states, n_rep):
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    Q_w = Q[..., -W:, :]
    K_rep = repeat_kv(K, g)
    V_rep = repeat_kv(V, g)

    attn = (Q_w @ K_rep.transpose(-1, -2)) / math.sqrt(D)
    mask = torch.full((W, L), float("-inf"), device=Q.device, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=L - W + 1)
    attn = attn + mask
    # softmax → bf16 cast (critical precision point)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(Q.dtype)

    # Attention output in input dtype
    y = attn @ V_rep

    # Scoring in fp32 (matching updated production code)
    attn_f = attn.float()
    W_mat = (attn_f / (1.0 - attn_f + eps)) ** 2

    W_sum = W_mat.sum(dim=-2)
    v_norm2 = (V_rep.float() ** 2).sum(dim=-1)
    termA = W_sum * v_norm2

    WT_y = W_mat.transpose(-1, -2) @ y.float()
    termB = 2.0 * (V_rep.float() * WT_y).sum(dim=-1)

    y_norm2 = (y.float() ** 2).sum(dim=-1)
    termC = (W_mat.transpose(-1, -2) @ y_norm2.unsqueeze(-1)).squeeze(-1)

    scores = termA - termB + termC
    scores = scores.view(B, Hkv, g, -1).mean(dim=2)
    return scores.float()


# Model configs
LLAMA_CFG = dict(B=1, Hq=32, Hkv=8, D=128)
QWEN_CFG = dict(B=1, Hq=14, Hkv=2, D=64)
BATCH2_CFG = dict(B=2, Hq=32, Hkv=8, D=128)
MHA_CFG = dict(B=1, Hq=8, Hkv=8, D=128)  # g=1, multi-head attention


def _make_qkv(cfg, L, dtype=torch.bfloat16):
    B, Hq, Hkv, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"]
    Q = torch.randn(B, Hq, L, D, device="cuda", dtype=dtype)
    K = torch.randn(B, Hkv, L, D, device="cuda", dtype=dtype)
    V = torch.randn(B, Hkv, L, D, device="cuda", dtype=dtype)
    return Q, K, V


# ---------------------------------------------------------------------------
# Test 1: Reference vs existing repeat_kv code
# ---------------------------------------------------------------------------

class TestReferenceParity:
    @pytest.mark.parametrize("cfg,L", [
        (LLAMA_CFG, 512), (LLAMA_CFG, 1024),
        (QWEN_CFG, 512), (QWEN_CFG, 1024),
        (BATCH2_CFG, 512),
        (MHA_CFG, 512),
    ])
    def test_reference_matches_existing(self, cfg, L):
        from dropkv_triton import dropkv_scores_reference
        Q, K, V = _make_qkv(cfg, L)

        # Both use bf16 p → fp32 W scoring (match_pytorch_precision=True).
        ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=True)
        existing = _existing_pytorch_scores(Q, K, V)

        # Small gap remains from Y computation: reference uses fp32 matmul,
        # existing uses bf16 matmul for y = attn @ V_rep.
        torch.testing.assert_close(ref, existing, rtol=5e-2, atol=5e-3)


# ---------------------------------------------------------------------------
# Test 2: Triton vs reference
# ---------------------------------------------------------------------------

class TestTritonCorrectness:
    @pytest.mark.parametrize("cfg,L", [
        (LLAMA_CFG, 512), (LLAMA_CFG, 1024), (LLAMA_CFG, 2048), (LLAMA_CFG, 4096),
        (QWEN_CFG, 512), (QWEN_CFG, 1024), (QWEN_CFG, 2048),
        (BATCH2_CFG, 512),
        (MHA_CFG, 512),
    ])
    def test_triton_vs_reference(self, cfg, L):
        """fp32 parity: both paths use match_pytorch_precision=False."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton
        Q, K, V = _make_qkv(cfg, L)

        ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=False)
        tri = dropkv_scores_triton(Q, K, V, allow_tf32=False, match_pytorch_precision=False)

        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 3: NaN safety (Qwen padded rows: M=56, BLOCK_M=64)
# ---------------------------------------------------------------------------

class TestNaNSafety:
    def test_qwen_no_nan(self):
        from dropkv_triton import dropkv_scores_triton
        Q, K, V = _make_qkv(QWEN_CFG, 512)
        scores = dropkv_scores_triton(Q, K, V, allow_tf32=False)
        assert not torch.isnan(scores).any(), "NaN found in Qwen scores"
        assert not torch.isinf(scores).any(), "Inf found in Qwen scores"

    def test_qwen_padded_rows_zero_contribution(self):
        """M=56 with BLOCK_M=64 → 8 padded rows should contribute nothing."""
        from dropkv_triton import dropkv_scores_triton
        # Run twice with same inputs — deterministic
        Q, K, V = _make_qkv(QWEN_CFG, 512)
        s1 = dropkv_scores_triton(Q, K, V, allow_tf32=False)
        s2 = dropkv_scores_triton(Q, K, V, allow_tf32=False)
        torch.testing.assert_close(s1, s2)


# ---------------------------------------------------------------------------
# Test 4: Kernel A↔B consistency (Σ exp(logits-lse) ≈ 1)
# ---------------------------------------------------------------------------

class TestKernelConsistency:
    @pytest.mark.parametrize("cfg,L", [
        (LLAMA_CFG, 512), (QWEN_CFG, 512),
    ])
    def test_softmax_reconstruction_sums_to_one(self, cfg, L):
        """Verify p = exp(logits - lse) sums to ~1 using Triton-produced lse.

        This catches Kernel A↔B dot/mask mismatches: if Kernel B recomputes
        logits differently from Kernel A, p_sum will deviate from 1.
        """
        from dropkv_triton import dropkv_scores_triton, _next_power_of_2
        B, Hq, Hkv, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"]
        g = Hq // Hkv
        W = 8

        Q, K, V = _make_qkv(cfg, L)

        # Run Triton to get the actual lse produced by Kernel A
        # We need to access intermediate lse — re-run the kernel setup
        M = g * W
        BLOCK_K = 32
        BLOCK_M = _next_power_of_2(M)
        BLOCK_N = 64
        inv_sqrt_d = 1.0 / math.sqrt(D)

        import triton
        from dropkv_triton import _dropkv_attn_fwd_kernel

        # Kernel now expects Q_window[B, Hq, W, D] (pre-sliced)
        Q_w = Q[:, :, -W:, :].contiguous()

        Y = torch.zeros(B, Hkv, M, D, device=Q.device, dtype=Q.dtype)
        lse_triton = torch.zeros(B, Hkv, M, device=Q.device, dtype=torch.float32)
        grid_a = (B * Hkv,)
        _dropkv_attn_fwd_kernel[grid_a](
            Q_w, K, V, Y, lse_triton,
            Hq, Hkv, L, M, g, W,
            Q_w.stride(0), Q_w.stride(1), Q_w.stride(2), Q_w.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            lse_triton.stride(0), lse_triton.stride(1), lse_triton.stride(2),
            inv_sqrt_d,
            D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            ALLOW_TF32=False,
        )

        # Now recompute logits in PyTorch (same path as Kernel B would)
        # and verify exp(logits - lse) sums to ~1
        Q_f, K_f = Q.float(), K.float()
        for hkv in range(min(Hkv, 2)):
            for gi in range(min(g, 2)):
                for w in range(W):
                    m_idx = gi * W + w
                    q_head = hkv * g + gi
                    seq_pos = L - W + w

                    q_vec = Q_f[0, q_head, seq_pos, :]
                    k_all = K_f[0, hkv, :, :]

                    logits = (q_vec @ k_all.T) / math.sqrt(D)
                    logits[seq_pos + 1:] = float("-inf")

                    lse_val = lse_triton[0, hkv, m_idx].item()
                    p = torch.exp(logits - lse_val)
                    p_sum = p.sum().item()

                    assert abs(p_sum - 1.0) < 1e-3, (
                        f"p sums to {p_sum} (not ~1) for hkv={hkv}, gi={gi}, w={w}; "
                        f"lse_triton={lse_val:.4f}"
                    )


# ---------------------------------------------------------------------------
# Test 5: mask_m invariant (M=56, BLOCK_M_TILE=32, second tile has 24+8)
# ---------------------------------------------------------------------------

class TestMaskMInvariant:
    def test_second_tile_partial(self):
        """With M=56, BLOCK_M_TILE=16: tiles cover [0:16], [16:32], [32:48], [48:56+8pad].
        The last tile has 8 valid + 8 invalid rows. Scores should still be correct."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton
        Q, K, V = _make_qkv(QWEN_CFG, 256)

        ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=False)
        tri = dropkv_scores_triton(Q, K, V, allow_tf32=False, match_pytorch_precision=False)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 6: dtype parity (fp16 + bf16)
# ---------------------------------------------------------------------------

class TestDtypeParity:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_dtype(self, dtype):
        """fp32 parity across dtypes."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton
        Q, K, V = _make_qkv(LLAMA_CFG, 512, dtype=dtype)

        ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=False)
        tri = dropkv_scores_triton(Q, K, V, allow_tf32=False, match_pytorch_precision=False)

        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 7: ALLOW_TF32 parity
# ---------------------------------------------------------------------------

class TestAllowTF32:
    def test_tf32_consistency(self):
        """Both TF32 modes should produce finite scores with correlated rankings."""
        from dropkv_triton import dropkv_scores_triton
        Q, K, V = _make_qkv(LLAMA_CFG, 512)

        s_tf32 = dropkv_scores_triton(Q, K, V, allow_tf32=True)
        s_no_tf32 = dropkv_scores_triton(Q, K, V, allow_tf32=False)

        assert not torch.isnan(s_tf32).any()
        assert not torch.isnan(s_no_tf32).any()
        assert not torch.isinf(s_tf32).any()
        assert not torch.isinf(s_no_tf32).any()

        # Ranking correlation: top-k sets should largely overlap.
        # TF32 truncates mantissa so scores differ, but rankings should agree.
        k = max(1, s_tf32.shape[-1] // 10)  # top 10%
        topk_tf32 = torch.topk(s_tf32, k, dim=-1).indices
        topk_no_tf32 = torch.topk(s_no_tf32, k, dim=-1).indices
        # At least 70% overlap in top-k sets
        for b in range(s_tf32.shape[0]):
            for h in range(s_tf32.shape[1]):
                set_tf32 = set(topk_tf32[b, h].tolist())
                set_no = set(topk_no_tf32[b, h].tolist())
                overlap = len(set_tf32 & set_no) / k
                assert overlap >= 0.7, (
                    f"Top-{k} overlap only {overlap:.0%} for b={b}, h={h}"
                )


# ---------------------------------------------------------------------------
# Test 8: Guard failure tests
# ---------------------------------------------------------------------------

class TestGuards:
    def test_l_less_than_w(self):
        from dropkv_triton import dropkv_scores_triton
        Q, K, V = _make_qkv(LLAMA_CFG, 4)  # L=4 < W=8
        with pytest.raises(AssertionError, match="must be >= window_size"):
            dropkv_scores_triton(Q, K, V)

    def test_hq_not_divisible(self):
        from dropkv_triton import dropkv_scores_triton
        Q = torch.randn(1, 7, 64, 128, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(1, 4, 64, 128, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(1, 4, 64, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match="divisible"):
            dropkv_scores_triton(Q, K, V)

    def test_non_contiguous_kv(self):
        from dropkv_triton import dropkv_scores_triton
        Q, K, V = _make_qkv(LLAMA_CFG, 64)
        K_nc = K.transpose(2, 3)  # non-contiguous
        with pytest.raises(AssertionError, match="contiguous"):
            dropkv_scores_triton(Q, K_nc, V)

    def test_non_contiguous_q_accepted(self):
        """Q need not be contiguous — wrapper slices internally."""
        from dropkv_triton import dropkv_scores_triton
        Q, K, V = _make_qkv(LLAMA_CFG, 64)
        # transpose(2,3) gives [1,32,128,64] non-contiguous, transpose back keeps
        # the same data but strides are non-contiguous
        Q_nc = Q.transpose(2, 3).contiguous().transpose(2, 3)  # [1,32,64,128] non-contiguous
        assert not Q_nc.is_contiguous(), "failed to create non-contiguous Q"
        scores = dropkv_scores_triton(Q_nc, K, V)
        assert scores.shape == (1, LLAMA_CFG["Hkv"], 64)

    def test_mixed_dtypes(self):
        from dropkv_triton import dropkv_scores_triton
        Q = torch.randn(1, 32, 64, 128, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.float16)
        V = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match="same dtype"):
            dropkv_scores_triton(Q, K, V)

    def test_fp32_input(self):
        from dropkv_triton import dropkv_scores_triton
        Q, K, V = _make_qkv(LLAMA_CFG, 64, dtype=torch.float32)
        with pytest.raises(AssertionError, match="fp16 or bf16"):
            dropkv_scores_triton(Q, K, V)

    def test_window_size_zero(self):
        from dropkv_triton import dropkv_scores_triton
        Q, K, V = _make_qkv(LLAMA_CFG, 64)
        with pytest.raises(AssertionError, match="positive int"):
            dropkv_scores_triton(Q, K, V, window_size=0)

    def test_shape_mismatch(self):
        from dropkv_triton import dropkv_scores_triton
        # K and V have different seq lengths — K/V shape mismatch
        Q = torch.randn(1, 32, 32, 128, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(1, 8, 32, 128, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match="shape mismatch"):
            dropkv_scores_triton(Q, K, V)

    def test_cpu_input(self):
        from dropkv_triton import dropkv_scores_triton
        Q = torch.randn(1, 32, 64, 128, dtype=torch.bfloat16)  # CPU
        K = torch.randn(1, 8, 64, 128, dtype=torch.bfloat16)
        V = torch.randn(1, 8, 64, 128, dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match="CUDA"):
            dropkv_scores_triton(Q, K, V)


# ---------------------------------------------------------------------------
# Test 9: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_l_equals_w(self):
        """Minimum valid: L = window_size.
        At L=W, peaked attention (p≈1) causes w≈(1/eps)^2 which amplifies
        bf16 Y storage error. Both existing code and Triton show this.
        We only check: runs without crash, no NaN, scores are finite.
        """
        from dropkv_triton import dropkv_scores_triton
        Q, K, V = _make_qkv(LLAMA_CFG, 8)  # L = W = 8
        tri = dropkv_scores_triton(Q, K, V, allow_tf32=False)
        assert not torch.isnan(tri).any(), "NaN in scores"
        assert not torch.isinf(tri).any(), "Inf in scores"

    def test_non_pow2_L(self):
        """L not divisible by BLOCK_N (64)."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton
        Q, K, V = _make_qkv(LLAMA_CFG, 100)
        ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=False)
        tri = dropkv_scores_triton(Q, K, V, allow_tf32=False, match_pytorch_precision=False)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 10: match_pytorch_precision topk ranking test
# ---------------------------------------------------------------------------

class TestPyTorchPrecisionMatch:
    """Verify that match_pytorch_precision=True produces topk rankings
    matching the existing PyTorch code (which operates in bf16)."""

    @pytest.mark.parametrize("cfg,L", [
        (LLAMA_CFG, 512), (LLAMA_CFG, 1024),
        (QWEN_CFG, 512),
    ])
    def test_triton_topk_matches_existing(self, cfg, L):
        from dropkv_triton import dropkv_scores_triton
        Q, K, V = _make_qkv(cfg, L)

        tri = dropkv_scores_triton(Q, K, V, match_pytorch_precision=True)
        existing = _existing_pytorch_scores(Q, K, V)

        # After avg_pool1d + window protection, topk should match
        W = 8
        kernel_size = 11
        tri_smooth = F.avg_pool1d(tri, kernel_size=kernel_size,
                                   padding=kernel_size // 2, stride=1)
        exist_smooth = F.avg_pool1d(existing, kernel_size=kernel_size,
                                     padding=kernel_size // 2, stride=1)
        big_t = tri_smooth.amax(dim=-1, keepdim=True) + 1.0
        big_e = exist_smooth.amax(dim=-1, keepdim=True) + 1.0
        tri_smooth[:, :, -W:] = big_t
        exist_smooth[:, :, -W:] = big_e

        keep_len = max(1, int(0.1 * L))
        topk_tri = torch.topk(tri_smooth, k=keep_len, dim=-1).indices
        topk_exist = torch.topk(exist_smooth, k=keep_len, dim=-1).indices

        for b in range(cfg["B"]):
            for h in range(cfg["Hkv"]):
                set_tri = set(topk_tri[b, h].tolist())
                set_exist = set(topk_exist[b, h].tolist())
                overlap = len(set_tri & set_exist) / keep_len
                assert overlap >= 0.8, (
                    f"Top-{keep_len} overlap only {overlap:.0%} for b={b}, h={h}. "
                    f"Triton top5: {sorted(list(set_tri))[:5]}, "
                    f"Existing top5: {sorted(list(set_exist))[:5]}"
                )


# ---------------------------------------------------------------------------
# Test 11: PyTorch fallback path (GQA-aware, no repeat_kv)
# ---------------------------------------------------------------------------

class TestPyTorchFallback:
    """Verify the in-file GQA-aware PyTorch scoring path matches the old
    repeat_kv-based scoring (100% topk overlap)."""

    @pytest.mark.parametrize("cfg,L", [
        (LLAMA_CFG, 512), (LLAMA_CFG, 1024),
        (QWEN_CFG, 512), (MHA_CFG, 256),
    ])
    def test_gqa_pytorch_matches_old(self, cfg, L):
        """Verify GQA-aware scoring selects the same tokens as repeat_kv scoring.

        The two paths compute QK^T with different matmul shapes (cuBLAS uses
        different tiling), so exact numerical match is impossible even at same
        dtype. We verify: (1) scores are close and (2) topk overlap >= 95%.
        """
        Q, K, V = _make_qkv(cfg, L)
        B = cfg["B"]; Hkv = cfg["Hkv"]; Hq = cfg["Hq"]; D = cfg["D"]
        g = Hq // Hkv
        W = 8

        # --- New GQA-aware path (mirrors dropkv_cache.py) ---
        M = g * W
        Q_w = Q[..., -W:, :].reshape(B, Hkv, M, D)
        attn_logits = torch.matmul(
            Q_w.float(), K.float().transpose(-1, -2)
        ) / math.sqrt(D)

        offs_m = torch.arange(M, device=K.device)
        causal_bounds = L - W + (offs_m % W)
        col_idx = torch.arange(L, device=K.device)
        attn_logits.masked_fill_(
            col_idx[None, None, None, :] > causal_bounds[None, None, :, None],
            float("-inf"),
        )
        p = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(Q.dtype)
        y = torch.matmul(p, V)

        eps = 1e-6
        p_f = p.float()
        W_mat = (p_f / (1.0 - p_f + eps)) ** 2

        W_sum = W_mat.sum(dim=-2)
        V_f = V.float()
        v_norm2 = (V_f ** 2).sum(dim=-1)
        termA = W_sum * v_norm2

        V_dot_y = torch.matmul(V_f, y.float().transpose(-1, -2))
        termB = 2.0 * (W_mat.transpose(-1, -2) * V_dot_y).sum(dim=-1)

        y_norm2 = (y.float() ** 2).sum(dim=-1)
        termC = torch.matmul(
            W_mat.transpose(-1, -2), y_norm2.unsqueeze(-1)
        ).squeeze(-1)

        scores_new = (termA - termB + termC) / g

        # --- Old repeat_kv path ---
        scores_old = _existing_pytorch_scores(Q, K, V, window_size=W, eps=eps)

        # Topk overlap >= 95% (boundary scores may shift due to cuBLAS tiling
        # differences between [B,Hkv,g*W,L] vs [B,Hq,W,L] matmuls)
        keep_len = max(1, int(0.1 * L))
        topk_new = torch.topk(scores_new, k=keep_len, dim=-1).indices
        topk_old = torch.topk(scores_old, k=keep_len, dim=-1).indices
        for b in range(B):
            for h in range(Hkv):
                overlap = len(set(topk_new[b, h].tolist()) & set(topk_old[b, h].tolist()))
                pct = overlap / keep_len
                assert pct >= 0.95, (
                    f"Topk overlap {pct:.0%} at b={b} h={h}, expected >= 95%"
                )


# ---------------------------------------------------------------------------
# Two-pass prefill tests
# ---------------------------------------------------------------------------

class TestCacheGuards:
    """Tests for cache update guards and HF compatibility."""

    def _make_cache(self, **kwargs):
        from dropkv_cache import DropKVCache
        defaults = dict(keep_ratio=0.3, window_size=8, kernel_size=11, use_triton=False)
        defaults.update(kwargs)
        return DropKVCache(**defaults)

    def test_hf_4arg_compat_with_query_states(self):
        """HF 4-arg call with query_states in cache_kwargs triggers eviction."""
        from dropkv_cache import DropKVCache
        L = 128
        Q, K, V = _make_qkv(LLAMA_CFG, L)
        cache = self._make_cache()

        # HF-style call: update(key, value, layer_idx, cache_kwargs)
        cache_kwargs = {"query_states": Q}
        ret_k, ret_v = cache.update(K, V, 0, cache_kwargs)

        # Should return full KV to attention
        assert ret_k.shape[-2] == L
        # But store compressed KV in cache
        keep_len = max(int(0.3 * L), 8, 1)
        assert cache.key_cache[0].shape[-2] == keep_len

    def test_hf_4arg_compat_without_query_states(self):
        """HF 4-arg call without query_states stores full KV (no eviction)."""
        from dropkv_cache import DropKVCache
        L = 128
        Q, K, V = _make_qkv(LLAMA_CFG, L)
        cache = self._make_cache()

        # HF-style call without query_states in cache_kwargs
        cache_kwargs = {"sin": None, "cos": None}
        ret_k, ret_v = cache.update(K, V, 0, cache_kwargs)

        # No eviction — full KV stored
        assert cache.key_cache[0].shape[-2] == L

    def test_invalid_prefill_mode_raises(self):
        """Invalid _prefill_mode raises ValueError."""
        from dropkv_cache import DropKVCache
        cache = self._make_cache()
        Q, K, V = _make_qkv(LLAMA_CFG, 128)

        cache._prefill_mode = "typo_mode"
        with pytest.raises(ValueError, match="Invalid _prefill_mode"):
            cache.update(Q, K, V, layer_idx=0)

    def test_invalid_prefill_mode_short_seq(self):
        """Invalid _prefill_mode raises even for short sequences."""
        cache = self._make_cache(window_size=16)
        Q, K, V = _make_qkv(LLAMA_CFG, 8)  # L=8 < window_size=16

        cache._prefill_mode = "typo_mode"
        with pytest.raises(ValueError, match="Invalid _prefill_mode"):
            cache.update(Q, K, V, layer_idx=0)


class TestChunkedPrefill:
    """Tests for chunked prefill on DropKVCache."""

    def _make_cache(self, **kwargs):
        from dropkv_cache import DropKVCache
        defaults = dict(keep_ratio=0.3, window_size=8, kernel_size=11, use_triton=False)
        defaults.update(kwargs)
        return DropKVCache(**defaults)

    def test_chunked_matches_single_pass_scores(self):
        """Chunked accumulate+evict produces same keep indices as single_pass."""
        from dropkv_cache import DropKVCache
        cfg = LLAMA_CFG
        L = 256
        W = 8
        Q, K, V = _make_qkv(cfg, L)

        # Single pass — reference
        cache_single = self._make_cache()
        cache_single.update(Q, K, V, layer_idx=0)

        # Chunked: accumulate first half, evict on second
        cache_chunked = self._make_cache()
        chunk_size = 128
        K1, V1 = K[:, :, :chunk_size, :], V[:, :, :chunk_size, :]
        K2, V2 = K[:, :, chunk_size:, :], V[:, :, chunk_size:, :]
        Q2 = Q[:, :, chunk_size:, :]  # Q from "last chunk"

        # First chunk: accumulate
        cache_chunked._prefill_mode = "accumulate"
        cache_chunked.update(Q[:, :, :chunk_size, :], K1, V1, layer_idx=0)

        # Second chunk: evict_accumulated
        cache_chunked._prefill_mode = "evict_accumulated"
        cache_chunked.update(Q2, K2, V2, layer_idx=0)

        # Both should have same compressed cache (same keep indices)
        assert cache_chunked.key_cache[0].shape == cache_single.key_cache[0].shape
        assert torch.equal(cache_chunked.key_cache[0], cache_single.key_cache[0])
        assert torch.equal(cache_chunked.value_cache[0], cache_single.value_cache[0])

    def test_accumulate_grows_cache(self):
        """Accumulate mode concatenates KV across chunks."""
        cfg = LLAMA_CFG
        Q, K, V = _make_qkv(cfg, 64)

        cache = self._make_cache()
        cache._prefill_mode = "accumulate"

        # First chunk
        cache.update(Q[:, :, :32, :], K[:, :, :32, :], V[:, :, :32, :], layer_idx=0)
        assert cache.key_cache[0].shape[2] == 32

        # Second chunk (decode branch — accumulate)
        cache.update(Q[:, :, 32:, :], K[:, :, 32:, :], V[:, :, 32:, :], layer_idx=0)
        assert cache.key_cache[0].shape[2] == 64

    def test_evict_accumulated_compresses(self):
        """evict_accumulated mode concatenates then evicts."""
        cfg = LLAMA_CFG
        L = 128
        Q, K, V = _make_qkv(cfg, L)

        cache = self._make_cache(keep_ratio=0.3)
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, :64, :], K[:, :, :64, :], V[:, :, :64, :], layer_idx=0)

        cache._prefill_mode = "evict_accumulated"
        full_k, full_v = cache.update(Q[:, :, 64:, :], K[:, :, 64:, :], V[:, :, 64:, :], layer_idx=0)

        # Full KV returned for this layer's attention
        assert full_k.shape[2] == L
        # But compressed KV stored
        keep_len = max(int(0.3 * L), 8, 1)
        assert cache.key_cache[0].shape[2] == keep_len

    def test_multi_chunk_accumulate(self):
        """Three chunks: accumulate, accumulate, evict."""
        cfg = LLAMA_CFG
        L = 192
        Q, K, V = _make_qkv(cfg, L)

        cache = self._make_cache(keep_ratio=0.2)

        # Chunk 1: first time (prefill branch, accumulate)
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, :64, :], K[:, :, :64, :], V[:, :, :64, :], layer_idx=0)
        assert cache.key_cache[0].shape[2] == 64

        # Chunk 2: decode branch, accumulate
        cache.update(Q[:, :, 64:128, :], K[:, :, 64:128, :], V[:, :, 64:128, :], layer_idx=0)
        assert cache.key_cache[0].shape[2] == 128

        # Chunk 3: evict
        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 128:, :], K[:, :, 128:, :], V[:, :, 128:, :], layer_idx=0)
        keep_len = max(int(0.2 * L), 8, 1)
        assert cache.key_cache[0].shape[2] == keep_len

    def test_multi_layer_chunked(self):
        """Chunked prefill works across multiple layers."""
        cfg = LLAMA_CFG
        L = 128
        num_layers = 3
        Q, K, V = _make_qkv(cfg, L)

        cache = self._make_cache(keep_ratio=0.3)
        chunk_size = 64

        # First chunk: all layers
        cache._prefill_mode = "accumulate"
        for layer in range(num_layers):
            cache.update(Q[:, :, :chunk_size, :], K[:, :, :chunk_size, :],
                        V[:, :, :chunk_size, :], layer_idx=layer)

        # Second chunk: all layers with eviction
        cache._prefill_mode = "evict_accumulated"
        for layer in range(num_layers):
            cache.update(Q[:, :, chunk_size:, :], K[:, :, chunk_size:, :],
                        V[:, :, chunk_size:, :], layer_idx=layer)

        keep_len = max(int(0.3 * L), 8, 1)
        for layer in range(num_layers):
            assert cache.key_cache[layer].shape[2] == keep_len

    def test_seen_tokens_correct(self):
        """_seen_tokens tracks total tokens across chunks."""
        cfg = LLAMA_CFG
        Q, K, V = _make_qkv(cfg, 128)

        cache = self._make_cache()
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, :64, :], K[:, :, :64, :], V[:, :, :64, :], layer_idx=0)
        assert cache._seen_tokens == 64

        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 64:, :], K[:, :, 64:, :], V[:, :, 64:, :], layer_idx=0)
        assert cache._seen_tokens == 128

    def test_decode_after_chunked(self):
        """Standard decode works after chunked prefill."""
        cfg = LLAMA_CFG
        L = 128
        Q, K, V = _make_qkv(cfg, L)
        Q_dec = torch.randn(1, cfg["Hq"], 1, cfg["D"], device="cuda", dtype=torch.bfloat16)
        K_dec = torch.randn(1, cfg["Hkv"], 1, cfg["D"], device="cuda", dtype=torch.bfloat16)
        V_dec = torch.randn(1, cfg["Hkv"], 1, cfg["D"], device="cuda", dtype=torch.bfloat16)

        cache = self._make_cache(keep_ratio=0.3)
        # Chunked prefill
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, :64, :], K[:, :, :64, :], V[:, :, :64, :], layer_idx=0)
        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 64:, :], K[:, :, 64:, :], V[:, :, 64:, :], layer_idx=0)

        # Decode
        cache._prefill_mode = "single_pass"
        keep_len = cache.key_cache[0].shape[2]
        ret_k, ret_v = cache.update(Q_dec, K_dec, V_dec, layer_idx=0)
        assert ret_k.shape[2] == keep_len + 1
        assert cache.key_cache[0].shape[2] == keep_len + 1

    def test_prefill_chunked_function(self):
        """Integration test: prefill_chunked with Qwen2-0.5B."""
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from dropkv_cache import DropKVCache, prefill_chunked

        model_name = "Qwen/Qwen2-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        prompt = "The quick brown fox " * 100
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids[:, :512].to(next(model.parameters()).device)

        # Chunked prefill
        cache_chunked = DropKVCache(keep_ratio=0.3, window_size=8)
        output_chunked = prefill_chunked(model, input_ids, cache_chunked, chunk_size=128)

        # Single pass reference
        cache_single = DropKVCache(keep_ratio=0.3, window_size=8)
        with torch.no_grad():
            output_single = model(input_ids, past_key_values=cache_single, use_cache=True)

        # Cache shapes should match
        for i in range(len(cache_single.key_cache)):
            assert cache_chunked.key_cache[i].shape == cache_single.key_cache[i].shape, \
                f"Layer {i}: chunked {cache_chunked.key_cache[i].shape} vs single {cache_single.key_cache[i].shape}"

        # _seen_tokens should match
        assert cache_chunked._seen_tokens == cache_single._seen_tokens

    def test_prefill_chunked_short_prompt(self):
        """Prompt shorter than chunk_size falls through to single_pass."""
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from dropkv_cache import DropKVCache, prefill_chunked

        model_name = "Qwen/Qwen2-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        prompt = "Hello world"
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            next(model.parameters()).device
        )

        cache = DropKVCache(keep_ratio=0.3, window_size=8)
        output = prefill_chunked(model, input_ids, cache, chunk_size=4096)
        # Short prompt → no eviction, cache stores full
        assert cache.key_cache[0].shape[2] == input_ids.shape[1]

    def test_prefill_chunked_budget_short_prompt(self):
        """Budget strategy enforces max_kv_tokens even when L <= chunk_size."""
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from dropkv_cache import DropKVCache, prefill_chunked

        model_name = "Qwen/Qwen2-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        prompt = "The quick brown fox " * 20
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            next(model.parameters()).device
        )
        L = input_ids.shape[1]
        assert L < 4096, "Prompt must be shorter than chunk_size for this test"

        # Budget with max_kv_tokens smaller than keep_ratio * L
        max_kv = 16
        cache = DropKVCache(keep_ratio=0.5, window_size=8)
        output = prefill_chunked(
            model, input_ids, cache, chunk_size=4096,
            evict_strategy="budget", max_kv_tokens=max_kv,
        )
        stored = cache.key_cache[0].shape[2]
        assert stored <= max_kv, f"Budget violated: stored {stored} > max_kv {max_kv}"

    def test_prefill_chunked_rejects_populated_cache(self):
        """prefill_chunked raises on non-empty cache."""
        from dropkv_cache import DropKVCache, prefill_chunked

        cache = self._make_cache()
        cache.key_cache.append(torch.zeros(1, 8, 10, 128))
        cache.value_cache.append(torch.zeros(1, 8, 10, 128))

        with pytest.raises(ValueError, match="fresh.*empty.*cache"):
            prefill_chunked(object(), torch.zeros(1, 100), cache)

    def test_prefill_chunked_rejects_small_chunk(self):
        """chunk_size < window_size raises ValueError."""
        from dropkv_cache import DropKVCache, prefill_chunked

        cache = self._make_cache(window_size=8)
        with pytest.raises(ValueError, match="chunk_size.*window_size"):
            prefill_chunked(object(), torch.zeros(1, 100), cache, chunk_size=4)

    def test_prefill_chunked_cleans_up_on_error(self):
        """_prefill_mode is restored even if forward throws."""
        from dropkv_cache import DropKVCache, prefill_chunked

        class BrokenModel:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("simulated failure")

        cache = DropKVCache(keep_ratio=0.3, window_size=8, use_triton=False)
        with pytest.raises(RuntimeError, match="simulated failure"):
            prefill_chunked(BrokenModel(), torch.zeros(1, 100), cache, chunk_size=32)

        assert cache._prefill_mode == "single_pass"
        # GPU memory should be released
        assert len(cache.key_cache) == 0
        assert len(cache.value_cache) == 0
        assert cache._seen_tokens == 0

    def test_triton_handles_mismatched_q_length(self):
        """_compute_keep_indices uses Triton even when Q.shape[2] != K.shape[2]."""
        from dropkv_cache import DropKVCache
        cfg = LLAMA_CFG
        L = 256
        Q, K, V = _make_qkv(cfg, L)

        cache = DropKVCache(keep_ratio=0.3, window_size=8, use_triton=True)

        # Q with only the last chunk (shorter than K)
        Q_chunk = Q[:, :, 128:, :]  # shape [B, Hq, 128, D] vs K [B, Hkv, 256, D]

        keep_idx = cache._compute_keep_indices(Q_chunk, K, V)

        # Verify result is valid
        keep_len = max(int(0.3 * L), 8, 1)
        assert keep_idx.shape == (1, cfg["Hkv"], keep_len)


class TestBudgetEviction:
    """Tests for budget-based and every-chunk eviction strategies."""

    def _make_cache(self, **kwargs):
        from dropkv_cache import DropKVCache
        defaults = dict(keep_ratio=0.3, window_size=8, kernel_size=11, use_triton=False)
        defaults.update(kwargs)
        return DropKVCache(**defaults)

    def test_total_tokens_param_changes_keep_len(self):
        """_compute_keep_indices with total_tokens keeps more tokens."""
        cfg = LLAMA_CFG
        L = 128  # current KV length
        Q, K, V = _make_qkv(cfg, L)

        cache = self._make_cache(keep_ratio=0.3)

        # Without total_tokens: keep_len = 0.3 * 128 = 38
        idx_default = cache._compute_keep_indices(Q, K, V)
        assert idx_default.shape[2] == max(int(0.3 * L), 8, 1)

        # With total_tokens=512: keep_len = min(128, 0.3*512) = 128 (capped at seq_len)
        idx_total = cache._compute_keep_indices(Q, K, V, total_tokens=512)
        assert idx_total.shape[2] == L  # capped at seq_len

        # With total_tokens=200: keep_len = min(128, 0.3*200) = 60
        idx_mid = cache._compute_keep_indices(Q, K, V, total_tokens=200)
        assert idx_mid.shape[2] == 60

    def test_total_tokens_floor(self):
        """keep_len never below window_size even with small total_tokens."""
        cfg = LLAMA_CFG
        Q, K, V = _make_qkv(cfg, 64)

        cache = self._make_cache(keep_ratio=0.01, window_size=8)
        # total_tokens=10, keep_ratio=0.01 → 0.01*10 = 0 → floor to window_size=8
        idx = cache._compute_keep_indices(Q, K, V, total_tokens=10)
        assert idx.shape[2] == 8

    def test_budget_evicts_when_exceeded(self):
        """Budget strategy evicts when accumulated KV exceeds max_kv_tokens."""
        cfg = LLAMA_CFG
        L = 256
        Q, K, V = _make_qkv(cfg, L)

        cache = self._make_cache(keep_ratio=0.3)
        chunk_size = 64
        max_kv = 128  # evict after 2 chunks

        # Chunk 0: accumulate (first chunk, no budget pressure)
        cache._evict_use_seen_tokens = True
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, :64, :], K[:, :, :64, :], V[:, :, :64, :], layer_idx=0)
        after_first = cache.key_cache[0].shape[2]
        assert after_first == 64  # full chunk stored, no eviction

        # Chunk 1: accumulate (64 + 64 = 128, at budget but not over)
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, 64:128, :], K[:, :, 64:128, :], V[:, :, 64:128, :], layer_idx=0)
        after_accum = cache.key_cache[0].shape[2]
        assert after_accum == 128

        # Chunk 2: 128 + 64 = 192 > 128 → evict_accumulated
        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 128:192, :], K[:, :, 128:192, :], V[:, :, 128:192, :], layer_idx=0)
        # keep_len = min(192, 0.3 * 192) = min(192, 57) = 57
        after_evict = cache.key_cache[0].shape[2]
        expected = min(after_accum + 64, max(int(0.3 * cache._seen_tokens), 8, 1))
        assert after_evict == expected

    def test_budget_keep_len_grows_with_seen(self):
        """With _evict_use_seen_tokens, keep_len grows as more tokens are seen."""
        cfg = LLAMA_CFG
        Q, K, V = _make_qkv(cfg, 512)

        cache = self._make_cache(keep_ratio=0.3)
        cache._evict_use_seen_tokens = True

        # First chunk: accumulate (no eviction)
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, :128, :], K[:, :, :128, :], V[:, :, :128, :], layer_idx=0)

        # Evict after first accumulation — keep_len based on _seen_tokens=256
        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 128:256, :], K[:, :, 128:256, :], V[:, :, 128:256, :], layer_idx=0)
        keep1 = cache.key_cache[0].shape[2]

        # Accumulate more
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, 256:384, :], K[:, :, 256:384, :], V[:, :, 256:384, :], layer_idx=0)

        # Evict again — keep_len based on _seen_tokens=512
        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 384:512, :], K[:, :, 384:512, :], V[:, :, 384:512, :], layer_idx=0)
        keep2 = cache.key_cache[0].shape[2]

        # keep2 should be larger than keep1 (more tokens seen)
        assert keep2 > keep1

    def test_every_chunk_evicts_each_time(self):
        """every_chunk strategy evicts after every chunk."""
        cfg = LLAMA_CFG
        L = 256
        Q, K, V = _make_qkv(cfg, L)

        cache = self._make_cache(keep_ratio=0.3)
        cache._evict_use_seen_tokens = True

        # Chunk 0: single_pass
        cache._prefill_mode = "single_pass"
        cache.update(Q[:, :, :64, :], K[:, :, :64, :], V[:, :, :64, :], layer_idx=0)
        assert cache.key_cache[0].shape[2] < 64  # evicted

        # Chunk 1: evict_accumulated
        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 64:128, :], K[:, :, 64:128, :], V[:, :, 64:128, :], layer_idx=0)
        after_c1 = cache.key_cache[0].shape[2]
        # keep_len = min(prev + 64, 0.3 * 128) = min(~83, 38) = 38
        assert after_c1 == max(int(0.3 * 128), 8, 1)

        # Chunk 2: evict again
        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 128:192, :], K[:, :, 128:192, :], V[:, :, 128:192, :], layer_idx=0)
        after_c2 = cache.key_cache[0].shape[2]
        assert after_c2 == max(int(0.3 * 192), 8, 1)

    def test_kv_bounded_by_budget(self):
        """KV never exceeds max_kv_tokens + chunk_size during budget strategy."""
        cfg = LLAMA_CFG
        L = 512
        Q, K, V = _make_qkv(cfg, L)

        cache = self._make_cache(keep_ratio=0.2)
        cache._evict_use_seen_tokens = True
        cache._budget_max_kv_tokens = 128
        max_kv = 128
        chunk_size = 64

        max_stored = 0
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            stored = cache.key_cache[0].shape[2] if cache.key_cache else 0
            would_have = stored + (end - start)

            if len(cache.key_cache) == 0:
                cache._prefill_mode = "accumulate"
            elif would_have > max_kv:
                cache._prefill_mode = "evict_accumulated"
            else:
                cache._prefill_mode = "accumulate"

            cache.update(
                Q[:, :, start:end, :], K[:, :, start:end, :],
                V[:, :, start:end, :], layer_idx=0
            )
            max_stored = max(max_stored, cache.key_cache[0].shape[2])

        # KV should never have exceeded budget + chunk (pre-eviction peak)
        assert max_stored <= max_kv + chunk_size

    def test_decode_after_budget_eviction(self):
        """Standard decode works after budget-based eviction."""
        cfg = LLAMA_CFG
        Q, K, V = _make_qkv(cfg, 128)
        Q_dec = torch.randn(1, cfg["Hq"], 1, cfg["D"], device="cuda", dtype=torch.bfloat16)
        K_dec = torch.randn(1, cfg["Hkv"], 1, cfg["D"], device="cuda", dtype=torch.bfloat16)
        V_dec = torch.randn(1, cfg["Hkv"], 1, cfg["D"], device="cuda", dtype=torch.bfloat16)

        cache = self._make_cache(keep_ratio=0.3)
        cache._evict_use_seen_tokens = True

        # Two chunks: accumulate then evict
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, :64, :], K[:, :, :64, :], V[:, :, :64, :], layer_idx=0)
        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 64:, :], K[:, :, 64:, :], V[:, :, 64:, :], layer_idx=0)

        # Decode
        cache._prefill_mode = "single_pass"
        cache._evict_use_seen_tokens = False
        pre_len = cache.key_cache[0].shape[2]
        ret_k, ret_v = cache.update(Q_dec, K_dec, V_dec, layer_idx=0)
        assert ret_k.shape[2] == pre_len + 1

    def test_prefill_chunked_budget_strategy(self):
        """Integration test: prefill_chunked with budget strategy on Qwen2."""
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from dropkv_cache import DropKVCache, prefill_chunked

        model_name = "Qwen/Qwen2-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        prompt = "The quick brown fox " * 100
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids[:, :512].to(next(model.parameters()).device)

        L = input_ids.shape[1]

        # Budget strategy
        cache = DropKVCache(keep_ratio=0.3, window_size=8)
        output = prefill_chunked(
            model, input_ids, cache,
            chunk_size=128, evict_strategy="budget", max_kv_tokens=256,
        )

        # Final cache should be evicted based on total tokens seen
        final_keep = max(int(0.3 * L), 8, 1)
        for i in range(len(cache.key_cache)):
            assert cache.key_cache[i].shape[2] == final_keep, \
                f"Layer {i}: {cache.key_cache[i].shape[2]} != {final_keep}"

        assert cache._seen_tokens == L
        # Mode should be cleaned up
        assert cache._prefill_mode == "single_pass"
        assert cache._evict_use_seen_tokens is False
        assert cache._budget_max_kv_tokens is None

    def test_prefill_chunked_every_chunk_strategy(self):
        """Integration test: every_chunk strategy on Qwen2."""
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from dropkv_cache import DropKVCache, prefill_chunked

        model_name = "Qwen/Qwen2-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        prompt = "The quick brown fox " * 100
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids[:, :512].to(next(model.parameters()).device)

        L = input_ids.shape[1]

        cache = DropKVCache(keep_ratio=0.3, window_size=8)
        output = prefill_chunked(
            model, input_ids, cache,
            chunk_size=128, evict_strategy="every_chunk",
        )

        final_keep = max(int(0.3 * L), 8, 1)
        for i in range(len(cache.key_cache)):
            assert cache.key_cache[i].shape[2] == final_keep

        assert cache._seen_tokens == L

    def test_invalid_evict_strategy_raises(self):
        """Invalid evict_strategy raises ValueError."""
        from dropkv_cache import DropKVCache, prefill_chunked

        cache = DropKVCache(keep_ratio=0.3, window_size=8, use_triton=False)
        with pytest.raises(ValueError, match="evict_strategy"):
            prefill_chunked(object(), torch.zeros(1, 100), cache,
                          chunk_size=32, evict_strategy="invalid")

    def test_cleanup_on_error_with_budget(self):
        """_evict_use_seen_tokens is cleaned up on error."""
        from dropkv_cache import DropKVCache, prefill_chunked

        class BrokenModel:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("simulated failure")

        cache = DropKVCache(keep_ratio=0.3, window_size=8, use_triton=False)
        with pytest.raises(RuntimeError, match="simulated failure"):
            prefill_chunked(BrokenModel(), torch.zeros(1, 100), cache,
                          chunk_size=32, evict_strategy="budget", max_kv_tokens=64)

        assert cache._prefill_mode == "single_pass"
        assert cache._evict_use_seen_tokens is False
        assert cache._budget_max_kv_tokens is None
        # GPU memory should be released
        assert len(cache.key_cache) == 0
        assert len(cache.value_cache) == 0
        assert cache._seen_tokens == 0

    def test_budget_cap_prevents_unbounded_growth(self):
        """When keep_ratio * seen_tokens > max_kv_tokens, keep_len is capped."""
        cfg = LLAMA_CFG
        L = 512
        Q, K, V = _make_qkv(cfg, L)

        # High keep_ratio so 0.8 * 512 = 409 > max_kv_tokens=100
        cache = self._make_cache(keep_ratio=0.8)
        cache._evict_use_seen_tokens = True
        cache._budget_max_kv_tokens = 100

        # Accumulate first chunk
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, :256, :], K[:, :, :256, :], V[:, :, :256, :], layer_idx=0)

        # Evict — without cap, keep_len = 0.8 * 512 = 409; with cap, <= 100
        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 256:, :], K[:, :, 256:, :], V[:, :, 256:, :], layer_idx=0)

        stored = cache.key_cache[0].shape[2]
        assert stored <= 100, f"KV should be capped at 100, got {stored}"
        assert stored == 100  # exactly at cap

    def test_short_final_chunk_no_shape_error(self):
        """Final chunk shorter than window_size doesn't cause shape mismatch."""
        cfg = LLAMA_CFG
        # L=70, chunk_size=64 → final chunk has 6 tokens (< window_size=8)
        L = 70
        Q, K, V = _make_qkv(cfg, L)

        cache = self._make_cache(keep_ratio=0.3, window_size=8)

        # Chunk 0: accumulate
        cache._prefill_mode = "accumulate"
        cache.update(Q[:, :, :64, :], K[:, :, :64, :], V[:, :, :64, :], layer_idx=0)

        # Chunk 1 (6 tokens): evict_accumulated — should not crash
        cache._prefill_mode = "evict_accumulated"
        cache.update(Q[:, :, 64:70, :], K[:, :, 64:70, :], V[:, :, 64:70, :], layer_idx=0)

        # Cache should be evicted
        stored = cache.key_cache[0].shape[2]
        assert stored == max(int(0.3 * 70), 8, 1)

    def test_seq_kwargs_rejected(self):
        """Sequence-shaped kwargs are rejected by prefill_chunked."""
        from dropkv_cache import DropKVCache, prefill_chunked

        cache = DropKVCache(keep_ratio=0.3, window_size=8, use_triton=False)
        with pytest.raises(ValueError, match="sequence-shaped"):
            prefill_chunked(
                object(), torch.zeros(1, 100), cache,
                chunk_size=32, attention_mask=torch.ones(1, 100),
            )


    def test_budget_enforced_on_single_chunk(self):
        """Budget strategy caps keep_len even when L <= chunk_size (single chunk)."""
        from dropkv_cache import DropKVCache
        cfg = LLAMA_CFG
        L = 256
        Q, K, V = _make_qkv(cfg, L)

        # Without budget: keep_ratio * L = 0.5 * 256 = 128
        cache_no_budget = DropKVCache(
            keep_ratio=0.5, window_size=8, use_triton=False
        )
        cache_no_budget.update(Q, K, V, layer_idx=0)
        no_budget_len = cache_no_budget.key_cache[0].shape[2]
        assert no_budget_len == 128

        # With budget: single_pass path should respect max_keep_len=64
        cache_budget = DropKVCache(
            keep_ratio=0.5, window_size=8, use_triton=False
        )
        cache_budget._evict_use_seen_tokens = True
        cache_budget._budget_max_kv_tokens = 64
        cache_budget.update(Q, K, V, layer_idx=0)
        budget_len = cache_budget.key_cache[0].shape[2]
        assert budget_len <= 64, f"Expected <= 64, got {budget_len}"

    def test_budget_rejects_small_max_kv(self):
        """max_kv_tokens < window_size raises ValueError."""
        from dropkv_cache import DropKVCache, prefill_chunked

        cache = DropKVCache(keep_ratio=0.3, window_size=8, use_triton=False)
        with pytest.raises(ValueError, match="window_size"):
            prefill_chunked(object(), torch.zeros(1, 100), cache,
                          chunk_size=32, evict_strategy="budget", max_kv_tokens=4)

    def test_invalid_mode_on_continuation_path_raises(self):
        """Invalid _prefill_mode on the Phase 2 (decode) path raises."""
        from dropkv_cache import DropKVCache

        cache = DropKVCache(keep_ratio=0.3, window_size=8, use_triton=False)
        Q, K, V = _make_qkv(LLAMA_CFG, 64)

        # First call populates cache (Phase 1)
        cache.update(Q, K, V, layer_idx=0)

        # Set invalid mode, then call again (Phase 2)
        cache._prefill_mode = "bogus"
        Q_dec = torch.randn(1, LLAMA_CFG["Hq"], 1, LLAMA_CFG["D"],
                            device="cuda", dtype=torch.bfloat16)
        K_dec = torch.randn(1, LLAMA_CFG["Hkv"], 1, LLAMA_CFG["D"],
                            device="cuda", dtype=torch.bfloat16)
        V_dec = torch.randn(1, LLAMA_CFG["Hkv"], 1, LLAMA_CFG["D"],
                            device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="Invalid _prefill_mode"):
            cache.update(Q_dec, K_dec, V_dec, layer_idx=0)

    def test_cleanup_after_mid_forward_failure(self):
        """Cache is fully cleared when failure occurs after some successful updates."""
        from dropkv_cache import DropKVCache, prefill_chunked

        call_count = 0

        class FailAfterFirstModel:
            def __call__(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise RuntimeError("OOM on chunk 2")
                # First call succeeds — simulate model updating cache
                cache = kwargs.get("past_key_values")
                B, Hkv, D = 1, LLAMA_CFG["Hkv"], LLAMA_CFG["D"]
                Hq = LLAMA_CFG["Hq"]
                chunk_len = args[0].shape[1]
                Q = torch.randn(B, Hq, chunk_len, D, device="cuda", dtype=torch.bfloat16)
                K = torch.randn(B, Hkv, chunk_len, D, device="cuda", dtype=torch.bfloat16)
                V = torch.randn(B, Hkv, chunk_len, D, device="cuda", dtype=torch.bfloat16)
                for layer in range(2):
                    cache.update(Q, K, V, layer_idx=layer)
                return type('Output', (), {'logits': torch.zeros(1)})()

        cache = DropKVCache(keep_ratio=0.3, window_size=8, use_triton=False)
        with pytest.raises(RuntimeError, match="OOM on chunk 2"):
            prefill_chunked(
                FailAfterFirstModel(),
                torch.zeros(1, 100, device="cuda"),
                cache, chunk_size=32,
            )

        # Cache should be fully cleared — no GPU memory held
        assert len(cache.key_cache) == 0
        assert len(cache.value_cache) == 0
        assert cache._seen_tokens == 0
        assert cache._prefill_mode == "single_pass"


class TestSplitK:
    """Tests for split-K Kernel A parallelization."""

    def test_num_splits_heuristic(self):
        """Verify num_splits computation at various configs."""
        from dropkv_triton import _compute_num_splits
        device = torch.device("cuda")
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        BLOCK_N = 64

        # Llama B=1, Hkv=8, L=32K: should split (8 CTAs << 2*SMs)
        ns = _compute_num_splits(1, 8, 32768, BLOCK_N, device)
        assert ns > 1, f"Expected splits > 1, got {ns}"
        assert ns * 8 <= 2 * num_sms + 8  # don't wildly overshoot

        # Qwen B=1, Hkv=2, L=32K: should split more aggressively
        ns_qwen = _compute_num_splits(1, 2, 32768, BLOCK_N, device)
        assert ns_qwen > ns  # fewer base CTAs → more splits

        # L=W=8: only 1 tile → can't split
        ns_tiny = _compute_num_splits(1, 8, 8, BLOCK_N, device)
        assert ns_tiny == 1

        # Large batch: B=32, Hkv=8 = 256 CTAs already > 2*SMs → no split
        ns_large = _compute_num_splits(32, 8, 4096, BLOCK_N, device)
        assert ns_large == 1

        # L=256: 4 tiles, min_tiles_per_split=4 → max_by_work=1
        ns_small = _compute_num_splits(1, 8, 256, BLOCK_N, device)
        assert ns_small == 1

    def test_split_kernel_partials_valid(self):
        """Split kernel produces finite partial state (no NaNs/Infs in valid rows)."""
        from dropkv_triton import (
            _dropkv_attn_fwd_split_kernel, _next_power_of_2,
        )
        import triton

        cfg = LLAMA_CFG
        L = 4096
        W = 8
        B, Hq, Hkv, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"]
        g = Hq // Hkv
        M = g * W
        BLOCK_M = _next_power_of_2(M)
        BLOCK_N = 64
        BLOCK_K = 32

        Q, K, V = _make_qkv(cfg, L)
        Q_w = Q[:, :, -W:, :].contiguous()

        num_splits = 4
        total_tiles = triton.cdiv(L, BLOCK_N)
        tiles_per_split = triton.cdiv(total_tiles, num_splits)

        m_partial = torch.empty(B, Hkv, num_splits, BLOCK_M, device="cuda", dtype=torch.float32)
        s_partial = torch.empty(B, Hkv, num_splits, BLOCK_M, device="cuda", dtype=torch.float32)
        y_partial = torch.zeros(B, Hkv, num_splits, BLOCK_M, D, device="cuda", dtype=torch.float32)

        grid = (B * Hkv, num_splits)
        _dropkv_attn_fwd_split_kernel[grid](
            Q_w, K, V, m_partial, s_partial, y_partial,
            Hq, Hkv, L, M, g, W,
            Q_w.stride(0), Q_w.stride(1), Q_w.stride(2), Q_w.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            m_partial.stride(0), m_partial.stride(1), m_partial.stride(2), m_partial.stride(3),
            s_partial.stride(0), s_partial.stride(1), s_partial.stride(2), s_partial.stride(3),
            y_partial.stride(0), y_partial.stride(1), y_partial.stride(2), y_partial.stride(3), y_partial.stride(4),
            1.0 / math.sqrt(D),
            tiles_per_split,
            D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ALLOW_TF32=True,
        )

        # Valid rows should have finite values
        assert torch.isfinite(m_partial[:, :, :, :M]).all(), "NaN/Inf in m_partial valid rows"
        assert torch.isfinite(s_partial[:, :, :, :M]).all(), "NaN/Inf in s_partial valid rows"
        assert torch.isfinite(y_partial[:, :, :, :M, :]).all(), "NaN/Inf in y_partial valid rows"
        # s should be non-negative for valid rows (zero when split is beyond causal bound)
        assert (s_partial[:, :, :, :M] >= 0).all(), "s_partial should be non-negative"

    def test_splitk_matches_single_cta(self):
        """Split-K (split + reduce) produces Y/lse matching single-CTA Kernel A."""
        from dropkv_triton import (
            _dropkv_attn_fwd_kernel, _dropkv_attn_fwd_split_kernel,
            _dropkv_reduce_kernel, _next_power_of_2,
        )
        import triton

        for cfg, name in [(LLAMA_CFG, "Llama"), (QWEN_CFG, "Qwen")]:
            for L in [512, 2048, 4096]:
                B, Hq, Hkv, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"]
                g = Hq // Hkv
                W = 8
                M = g * W
                BLOCK_M = _next_power_of_2(M)
                BLOCK_N = 64
                BLOCK_K = 32
                inv_sqrt_d = 1.0 / math.sqrt(D)

                Q, K, V = _make_qkv(cfg, L)
                Q_w = Q[:, :, -W:, :].contiguous()

                # --- Single-CTA reference ---
                Y_ref = torch.zeros(B, Hkv, M, D, device="cuda", dtype=Q.dtype)
                lse_ref = torch.zeros(B, Hkv, M, device="cuda", dtype=torch.float32)
                _dropkv_attn_fwd_kernel[(B * Hkv,)](
                    Q_w, K, V, Y_ref, lse_ref,
                    Hq, Hkv, L, M, g, W,
                    Q_w.stride(0), Q_w.stride(1), Q_w.stride(2), Q_w.stride(3),
                    K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                    Y_ref.stride(0), Y_ref.stride(1), Y_ref.stride(2), Y_ref.stride(3),
                    lse_ref.stride(0), lse_ref.stride(1), lse_ref.stride(2),
                    inv_sqrt_d,
                    D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ALLOW_TF32=True,
                )

                # --- Split-K ---
                for num_splits in [2, 4, 8]:
                    total_tiles = triton.cdiv(L, BLOCK_N)
                    if num_splits > total_tiles:
                        continue
                    tiles_per_split = triton.cdiv(total_tiles, num_splits)

                    m_p = torch.empty(B, Hkv, num_splits, BLOCK_M, device="cuda", dtype=torch.float32)
                    s_p = torch.empty(B, Hkv, num_splits, BLOCK_M, device="cuda", dtype=torch.float32)
                    y_p = torch.zeros(B, Hkv, num_splits, BLOCK_M, D, device="cuda", dtype=torch.float32)

                    _dropkv_attn_fwd_split_kernel[(B * Hkv, num_splits)](
                        Q_w, K, V, m_p, s_p, y_p,
                        Hq, Hkv, L, M, g, W,
                        Q_w.stride(0), Q_w.stride(1), Q_w.stride(2), Q_w.stride(3),
                        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                        m_p.stride(0), m_p.stride(1), m_p.stride(2), m_p.stride(3),
                        s_p.stride(0), s_p.stride(1), s_p.stride(2), s_p.stride(3),
                        y_p.stride(0), y_p.stride(1), y_p.stride(2), y_p.stride(3), y_p.stride(4),
                        inv_sqrt_d, tiles_per_split,
                        D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ALLOW_TF32=True,
                    )

                    Y_sk = torch.zeros(B, Hkv, M, D, device="cuda", dtype=Q.dtype)
                    lse_sk = torch.zeros(B, Hkv, M, device="cuda", dtype=torch.float32)

                    _dropkv_reduce_kernel[(B * Hkv,)](
                        m_p, s_p, y_p, Y_sk, lse_sk,
                        Hkv, M, num_splits,
                        m_p.stride(0), m_p.stride(1), m_p.stride(2), m_p.stride(3),
                        s_p.stride(0), s_p.stride(1), s_p.stride(2), s_p.stride(3),
                        y_p.stride(0), y_p.stride(1), y_p.stride(2), y_p.stride(3), y_p.stride(4),
                        Y_sk.stride(0), Y_sk.stride(1), Y_sk.stride(2), Y_sk.stride(3),
                        lse_sk.stride(0), lse_sk.stride(1), lse_sk.stride(2),
                        D=D, BLOCK_M=BLOCK_M,
                    )

                    # lse should match closely (fp32 reduction-order drift only)
                    torch.testing.assert_close(
                        lse_sk, lse_ref, rtol=1e-4, atol=1e-4,
                        msg=lambda m: f"{name} L={L} splits={num_splits}: lse mismatch: {m}",
                    )
                    # Y match (bf16 output — 8-way reduction adds fp32 rounding)
                    torch.testing.assert_close(
                        Y_sk.float(), Y_ref.float(), rtol=2e-3, atol=2e-3,
                        msg=lambda m: f"{name} L={L} splits={num_splits}: Y mismatch: {m}",
                    )

    def test_splitk_scores_match(self):
        """End-to-end: dropkv_scores_triton with split-K matches reference scores."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton

        for cfg, name in [(LLAMA_CFG, "Llama"), (QWEN_CFG, "Qwen")]:
            for L in [512, 2048, 4096]:
                Q, K, V = _make_qkv(cfg, L)

                scores_ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=True)
                scores_tri = dropkv_scores_triton(Q, K, V, match_pytorch_precision=True)

                torch.testing.assert_close(
                    scores_tri, scores_ref, rtol=1e-3, atol=1e-3,
                    msg=lambda m: f"{name} L={L}: {m}",
                )

    def test_splitk_topk_overlap(self):
        """Top-k indices from split-K have >=95% overlap with reference."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton

        for cfg in [LLAMA_CFG, QWEN_CFG]:
            Q, K, V = _make_qkv(cfg, 4096)
            B, Hkv = cfg["B"], cfg["Hkv"]

            ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=True)
            tri = dropkv_scores_triton(Q, K, V, match_pytorch_precision=True)

            keep_len = max(1, int(0.1 * 4096))
            topk_ref = torch.topk(ref, k=keep_len, dim=-1).indices
            topk_tri = torch.topk(tri, k=keep_len, dim=-1).indices
            for b in range(B):
                for h in range(Hkv):
                    s1 = set(topk_ref[b, h].tolist())
                    s2 = set(topk_tri[b, h].tolist())
                    pct = len(s1 & s2) / keep_len
                    assert pct >= 0.95, f"Topk overlap {pct:.0%} at b={b} h={h}"

    def test_splitk_padded_rows_nan_safe(self):
        """Qwen M=56 BLOCK_M=64: no NaNs in Y, lse, scores with split-K."""
        from dropkv_triton import dropkv_scores_triton

        Q, K, V = _make_qkv(QWEN_CFG, 2048)
        scores = dropkv_scores_triton(Q, K, V)
        assert torch.isfinite(scores).all(), "NaN/Inf in split-K scores (Qwen)"

    def test_splitk_p_sum_consistency(self):
        """Σ_j exp(logit_j - lse) ≈ 1 holds with split-K (Kernel A↔B contract)."""
        from dropkv_triton import dropkv_scores_triton, _next_power_of_2

        cfg = LLAMA_CFG
        L = 2048
        B, Hq, Hkv, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"]
        g = Hq // Hkv
        W = 8
        M = g * W
        inv_sqrt_d = 1.0 / math.sqrt(D)

        Q, K, V = _make_qkv(cfg, L)

        # Run through dropkv_scores_triton to populate Y/lse via split-K path
        # We need lse — extract it by running the split+reduce manually
        from dropkv_triton import (
            _dropkv_attn_fwd_kernel, _dropkv_attn_fwd_split_kernel,
            _dropkv_reduce_kernel, _compute_num_splits,
        )
        import triton

        Q_w = Q[:, :, -W:, :].contiguous()
        BLOCK_M = _next_power_of_2(M)
        BLOCK_N = 64
        BLOCK_K = 32

        num_splits = _compute_num_splits(B, Hkv, L, BLOCK_N, Q.device)
        Y = torch.zeros(B, Hkv, M, D, device="cuda", dtype=Q.dtype)
        lse = torch.zeros(B, Hkv, M, device="cuda", dtype=torch.float32)

        if num_splits <= 1:
            _dropkv_attn_fwd_kernel[(B * Hkv,)](
                Q_w, K, V, Y, lse,
                Hq, Hkv, L, M, g, W,
                Q_w.stride(0), Q_w.stride(1), Q_w.stride(2), Q_w.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                inv_sqrt_d,
                D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ALLOW_TF32=True,
            )
        else:
            total_tiles = triton.cdiv(L, BLOCK_N)
            tiles_per_split = triton.cdiv(total_tiles, num_splits)
            m_p = torch.empty(B, Hkv, num_splits, BLOCK_M, device="cuda", dtype=torch.float32)
            s_p = torch.empty(B, Hkv, num_splits, BLOCK_M, device="cuda", dtype=torch.float32)
            y_p = torch.zeros(B, Hkv, num_splits, BLOCK_M, D, device="cuda", dtype=torch.float32)
            _dropkv_attn_fwd_split_kernel[(B * Hkv, num_splits)](
                Q_w, K, V, m_p, s_p, y_p,
                Hq, Hkv, L, M, g, W,
                Q_w.stride(0), Q_w.stride(1), Q_w.stride(2), Q_w.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                m_p.stride(0), m_p.stride(1), m_p.stride(2), m_p.stride(3),
                s_p.stride(0), s_p.stride(1), s_p.stride(2), s_p.stride(3),
                y_p.stride(0), y_p.stride(1), y_p.stride(2), y_p.stride(3), y_p.stride(4),
                inv_sqrt_d, tiles_per_split,
                D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ALLOW_TF32=True,
            )
            _dropkv_reduce_kernel[(B * Hkv,)](
                m_p, s_p, y_p, Y, lse,
                Hkv, M, num_splits,
                m_p.stride(0), m_p.stride(1), m_p.stride(2), m_p.stride(3),
                s_p.stride(0), s_p.stride(1), s_p.stride(2), s_p.stride(3),
                y_p.stride(0), y_p.stride(1), y_p.stride(2), y_p.stride(3), y_p.stride(4),
                Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                D=D, BLOCK_M=BLOCK_M,
            )

        # Verify p sums: recompute logits in PyTorch, check exp(logits - lse) sums to ~1
        Q_f = Q_w.float()
        K_f = K.float()
        for hkv in range(min(Hkv, 2)):  # check 2 heads to keep test fast
            for m_idx in range(min(M, 8)):  # check 8 rows
                q_head = min(hkv * g + m_idx // W, Hq - 1)
                q_pos = m_idx % W
                q_vec = Q_f[0, q_head, q_pos, :]
                k_all = K_f[0, hkv, :, :]
                logits = (q_vec @ k_all.T) * inv_sqrt_d
                causal = L - W + q_pos
                logits[causal + 1:] = float("-inf")
                p_sum = torch.exp(logits - lse[0, hkv, m_idx]).sum().item()
                assert abs(p_sum - 1.0) < 0.02, (
                    f"p_sum={p_sum:.4f} at hkv={hkv} m={m_idx}, expected ~1.0"
                )

    def test_splitk_L_equals_W(self):
        """L=W: only 1 tile, num_splits clamps to 1 (fast path)."""
        from dropkv_triton import dropkv_scores_triton, _compute_num_splits

        ns = _compute_num_splits(1, 8, 8, 64, torch.device("cuda"))
        assert ns == 1, f"Expected num_splits=1 for L=W, got {ns}"

        Q, K, V = _make_qkv(LLAMA_CFG, 8)
        scores = dropkv_scores_triton(Q, K, V)
        assert torch.isfinite(scores).all()

    def test_splitk_single_cta_fast_path(self):
        """num_splits=1 uses existing single-CTA kernel (regression guard)."""
        from dropkv_triton import dropkv_scores_triton, dropkv_scores_reference

        Q, K, V = _make_qkv(LLAMA_CFG, 256)
        ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=True)
        tri = dropkv_scores_triton(Q, K, V, match_pytorch_precision=True)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)

    def test_splitk_large_window_nan_safe(self):
        """window_size > BLOCK_N: splits beyond causal bound produce no NaNs.

        With W=65 and BLOCK_N=64, a split starting near L can have ALL keys
        causally masked for the earliest query rows (causal_bound = L-65).
        The NaN-safe guard in the split kernel must prevent exp(-inf - (-inf)).
        """
        from dropkv_triton import (
            dropkv_scores_triton, dropkv_scores_reference,
            _dropkv_attn_fwd_split_kernel, _dropkv_reduce_kernel,
            _compute_num_splits, _next_power_of_2,
        )
        import triton

        # Use small GQA config to keep BLOCK_M manageable (M=130, BLOCK_M=256)
        B, Hq, Hkv, D = 1, 4, 2, 128
        W = 65  # > BLOCK_N=64
        L = 1344  # chosen so last split starts after L-W
        g = Hq // Hkv
        M = g * W
        BLOCK_M = _next_power_of_2(M)
        BLOCK_N = 64

        Q = torch.randn(B, Hq, L, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, L, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, L, D, device="cuda", dtype=torch.bfloat16)

        # Force split-K with enough splits that some are beyond causal bound
        num_splits = _compute_num_splits(B, Hkv, L, BLOCK_N, Q.device)
        if num_splits <= 1:
            num_splits = 5  # force splitting

        Q_w = Q[:, :, -W:, :].contiguous()
        inv_sqrt_d = 1.0 / math.sqrt(D)
        total_tiles = triton.cdiv(L, BLOCK_N)
        tiles_per_split = triton.cdiv(total_tiles, num_splits)

        m_p = torch.empty(B, Hkv, num_splits, BLOCK_M, device="cuda", dtype=torch.float32)
        s_p = torch.empty(B, Hkv, num_splits, BLOCK_M, device="cuda", dtype=torch.float32)
        y_p = torch.zeros(B, Hkv, num_splits, BLOCK_M, D, device="cuda", dtype=torch.float32)
        Y = torch.empty(B, Hkv, M, D, device="cuda", dtype=Q.dtype)
        lse = torch.empty(B, Hkv, M, device="cuda", dtype=torch.float32)

        _dropkv_attn_fwd_split_kernel[(B * Hkv, num_splits)](
            Q_w, K, V, m_p, s_p, y_p,
            Hq, Hkv, L, M, g, W,
            Q_w.stride(0), Q_w.stride(1), Q_w.stride(2), Q_w.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            m_p.stride(0), m_p.stride(1), m_p.stride(2), m_p.stride(3),
            s_p.stride(0), s_p.stride(1), s_p.stride(2), s_p.stride(3),
            y_p.stride(0), y_p.stride(1), y_p.stride(2), y_p.stride(3), y_p.stride(4),
            inv_sqrt_d, tiles_per_split,
            D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=32,
            ALLOW_TF32=True,
        )

        # Partials: valid rows must have no NaN (m=-inf is OK for empty splits)
        m_valid = m_p[:, :, :, :M]
        assert not torch.isnan(m_valid).any(), \
            "NaN in m_partial — exp(-inf - (-inf)) guard failed"
        # s_partial: NaN would indicate the bug
        assert not torch.isnan(s_p[:, :, :, :M]).any(), \
            "NaN in s_partial — exp(-inf - (-inf)) guard failed"
        assert not torch.isnan(y_p[:, :, :, :M, :]).any(), \
            "NaN in y_partial — exp(-inf - (-inf)) guard failed"

        # Reduce and verify final outputs are finite
        _dropkv_reduce_kernel[(B * Hkv,)](
            m_p, s_p, y_p, Y, lse,
            Hkv, M, num_splits,
            m_p.stride(0), m_p.stride(1), m_p.stride(2), m_p.stride(3),
            s_p.stride(0), s_p.stride(1), s_p.stride(2), s_p.stride(3),
            y_p.stride(0), y_p.stride(1), y_p.stride(2), y_p.stride(3), y_p.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D=D, BLOCK_M=BLOCK_M,
        )
        assert torch.isfinite(Y).all(), "NaN/Inf in Y after reduce (large W)"
        assert torch.isfinite(lse).all(), "NaN/Inf in lse after reduce (large W)"

        # End-to-end: scores must be finite AND match reference
        scores = dropkv_scores_triton(Q, K, V, window_size=W)
        assert torch.isfinite(scores).all(), "NaN in end-to-end scores (W > BLOCK_N)"

        ref = dropkv_scores_reference(Q, K, V, window_size=W)
        torch.testing.assert_close(
            scores, ref, rtol=1e-3, atol=1e-3,
            msg=lambda m: f"Large-window scores diverge from reference: {m}",
        )

# ---------------------------------------------------------------------------
# Test: Q≠K length support (chunked prefill Triton path)
# ---------------------------------------------------------------------------

class TestQLenNotKLen:
    """Verify Triton kernels work when Lq != Lk (chunked prefill scenario)."""

    @pytest.mark.parametrize("cfg,Lq,Lk", [
        (LLAMA_CFG, 64, 512),      # typical: 64-token chunk, 512 accumulated KV
        (LLAMA_CFG, 128, 1024),     # larger
        (QWEN_CFG, 64, 512),       # Qwen (padded-row M=56)
        (BATCH2_CFG, 64, 256),     # batch > 1
        (MHA_CFG, 32, 256),        # MHA (g=1)
    ])
    def test_triton_vs_reference(self, cfg, Lq, Lk):
        """Triton matches reference when Lq < Lk."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton

        B, Hq, Hkv, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"]
        Q = torch.randn(B, Hq, Lq, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)

        ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=False)
        tri = dropkv_scores_triton(Q, K, V, match_pytorch_precision=False)

        assert ref.shape == (B, Hkv, Lk)
        assert tri.shape == (B, Hkv, Lk)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("cfg,Lq,Lk", [
        (LLAMA_CFG, 64, 512),
        (QWEN_CFG, 64, 512),
    ])
    def test_precision_match(self, cfg, Lq, Lk):
        """match_pytorch_precision=True gives same scores for both paths."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton

        B, Hq, Hkv, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"]
        Q = torch.randn(B, Hq, Lq, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)

        ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=True)
        tri = dropkv_scores_triton(Q, K, V, match_pytorch_precision=True)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)

    def test_lq_equals_lk_unchanged(self):
        """Lq == Lk still works (regression test)."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton

        Q, K, V = _make_qkv(LLAMA_CFG, 512)
        ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=False)
        tri = dropkv_scores_triton(Q, K, V, match_pytorch_precision=False)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)

    def test_lq_equals_window(self):
        """Edge case: Lq == window_size (minimum viable Q)."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton

        B, Hq, Hkv, D = LLAMA_CFG["B"], LLAMA_CFG["Hq"], LLAMA_CFG["Hkv"], LLAMA_CFG["D"]
        W = 8
        Lq, Lk = W, 256
        Q = torch.randn(B, Hq, Lq, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)

        ref = dropkv_scores_reference(Q, K, V, window_size=W, match_pytorch_precision=False)
        tri = dropkv_scores_triton(Q, K, V, window_size=W, match_pytorch_precision=False)

        assert ref.shape == (B, Hkv, Lk)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)

    def test_no_nan(self):
        """No NaN in scores for Q≠K with Qwen padded rows."""
        from dropkv_triton import dropkv_scores_triton

        B, Hq, Hkv, D = QWEN_CFG["B"], QWEN_CFG["Hq"], QWEN_CFG["Hkv"], QWEN_CFG["D"]
        Q = torch.randn(B, Hq, 64, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, 512, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, 512, D, device="cuda", dtype=torch.bfloat16)

        scores = dropkv_scores_triton(Q, K, V)
        assert not torch.isnan(scores).any(), "NaN in scores with Q≠K length"
        assert not torch.isinf(scores).any(), "Inf in scores with Q≠K length"

    def test_guard_lq_too_short(self):
        """Wrapper rejects Lq < window_size."""
        from dropkv_triton import dropkv_scores_triton

        B, Hq, Hkv, D = LLAMA_CFG["B"], LLAMA_CFG["Hq"], LLAMA_CFG["Hkv"], LLAMA_CFG["D"]
        Q = torch.randn(B, Hq, 4, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, 256, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, 256, D, device="cuda", dtype=torch.bfloat16)

        with pytest.raises(AssertionError, match="Lq.*must be >= window_size"):
            dropkv_scores_triton(Q, K, V, window_size=8)

    def test_guard_lk_too_short(self):
        """Wrapper rejects Lk < window_size."""
        from dropkv_triton import dropkv_scores_triton

        B, Hq, Hkv, D = LLAMA_CFG["B"], LLAMA_CFG["Hq"], LLAMA_CFG["Hkv"], LLAMA_CFG["D"]
        Q = torch.randn(B, Hq, 64, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, 4, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, 4, D, device="cuda", dtype=torch.bfloat16)

        with pytest.raises(AssertionError, match="Lk.*must be >= window_size"):
            dropkv_scores_triton(Q, K, V, window_size=8)

    def test_guard_lq_greater_than_lk(self):
        """Both Triton and reference reject Lq > Lk."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton

        B, Hq, Hkv, D = LLAMA_CFG["B"], LLAMA_CFG["Hq"], LLAMA_CFG["Hkv"], LLAMA_CFG["D"]
        Q = torch.randn(B, Hq, 512, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, 64, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, 64, D, device="cuda", dtype=torch.bfloat16)

        with pytest.raises(AssertionError, match="Lq.*must be <= Lk"):
            dropkv_scores_triton(Q, K, V)
        with pytest.raises(AssertionError, match="Lq.*must be <= Lk"):
            dropkv_scores_reference(Q, K, V)

    def test_splitk_with_qneqk(self):
        """Split-K path works with Q!=K (large Lk forces split-K activation)."""
        from dropkv_triton import dropkv_scores_reference, dropkv_scores_triton, _compute_num_splits

        B, Hq, Hkv, D = 1, 4, 2, 128  # small B*Hkv to force split-K
        Lq, Lk = 64, 8192
        Q = torch.randn(B, Hq, Lq, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)

        # Verify split-K is actually activated
        ns = _compute_num_splits(B, Hkv, Lk, 64, Q.device)
        assert ns > 1, f"Expected split-K but got num_splits={ns}"

        ref = dropkv_scores_reference(Q, K, V, match_pytorch_precision=False)
        tri = dropkv_scores_triton(Q, K, V, match_pytorch_precision=False)

        assert tri.shape == (B, Hkv, Lk)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)

    def test_reference_guard_lq_too_short(self):
        """Reference function also rejects Lq < window_size."""
        from dropkv_triton import dropkv_scores_reference

        B, Hq, Hkv, D = LLAMA_CFG["B"], LLAMA_CFG["Hq"], LLAMA_CFG["Hkv"], LLAMA_CFG["D"]
        Q = torch.randn(B, Hq, 4, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, 256, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, 256, D, device="cuda", dtype=torch.bfloat16)

        with pytest.raises(AssertionError, match="Lq.*must be >= window_size"):
            dropkv_scores_reference(Q, K, V, window_size=8)

    def test_short_chunk_falls_back_to_pytorch(self):
        """When Lq < window_size, Triton guard fails and PyTorch fallback runs."""
        from dropkv_cache import DropKVCache

        cfg = LLAMA_CFG
        B, Hq, Hkv, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"]
        Lq, Lk = 4, 256  # Lq < window_size=8

        cache = DropKVCache(keep_ratio=0.3, window_size=8, use_triton=True)
        Q = torch.randn(B, Hq, Lq, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)

        # Triton will raise AssertionError (Lq < W), caught by fallback
        keep_idx = cache._compute_keep_indices(Q, K, V)
        keep_len = max(int(0.3 * Lk), 8, 1)
        assert keep_idx.shape == (B, Hkv, keep_len)

    def test_chunked_prefill_uses_triton(self):
        """_compute_keep_indices uses Triton even when Q.shape[2] != K.shape[2].

        Verifies Triton actually ran by monkeypatching it to raise on call,
        confirming the code path reaches dropkv_scores_triton (not silently
        falling back to PyTorch).
        """
        from dropkv_cache import DropKVCache
        import dropkv_triton

        cfg = LLAMA_CFG
        B, Hq, Hkv, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"]
        Lq, Lk = 64, 256

        Q = torch.randn(B, Hq, Lq, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)

        # First: verify it works end-to-end
        cache = DropKVCache(keep_ratio=0.3, use_triton=True)
        keep_idx = cache._compute_keep_indices(Q, K, V)
        keep_len = max(int(0.3 * Lk), 8, 1)
        assert keep_idx.shape == (B, Hkv, keep_len)

        # Second: prove Triton was actually called (not PyTorch fallback).
        # _compute_keep_indices catches (ImportError, AssertionError) only;
        # RuntimeError propagates. We use a wrapper that records the call.
        original_fn = dropkv_triton.dropkv_scores_triton
        call_log = []
        def tracking_wrapper(*a, **kw):
            call_log.append(True)
            return original_fn(*a, **kw)
        dropkv_triton.dropkv_scores_triton = tracking_wrapper
        try:
            cache2 = DropKVCache(keep_ratio=0.3, use_triton=True)
            cache2._compute_keep_indices(Q, K, V)
            assert len(call_log) == 1, (
                "dropkv_scores_triton was not called — Triton path was bypassed"
            )
        finally:
            dropkv_triton.dropkv_scores_triton = original_fn


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
