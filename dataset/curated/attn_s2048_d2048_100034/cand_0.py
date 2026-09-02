import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100034
S, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _scale_softmax_kernel(
    S_ptr,            # scores (bf16), in/out
    n_cols,
    sqrt_d,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    ptr = S_ptr + row.to(tl.int64) * n_cols + offs
    s = tl.load(ptr, mask=mask, other=float('-inf'))
    # replicate: scores_bf16 = (q@k.T)/sqrt(d)  (division, rounded to bf16),
    # then softmax computed in fp32 (PyTorch upcasts bf16 softmax internally)
    t = (s.to(tl.float32) / sqrt_d).to(tl.bfloat16).to(tl.float32)
    t = tl.where(mask, t, float('-inf'))
    m = tl.max(t, axis=0)
    e = tl.exp(t - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.bfloat16)
    tl.store(ptr, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def _get_fused_weight(self, device, dtype):
        w = self._Wqkv
        if w is None or w.device != device or w.dtype != dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=device, dtype=dtype).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = x.shape[-1]

        if not x.is_cuda:
            # CPU fallback: reference path
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v

        Wqkv = self._get_fused_weight(x.device, x.dtype)

        # Single fused GEMM for Q, K, V projections
        qkv = x @ Wqkv
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        # Attention scores (bf16 GEMM via cuBLAS)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n = scores.shape[-1]
        m_rows = scores.numel() // n
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4

        _scale_softmax_kernel[(m_rows,)](
            scores, n, math.sqrt(d),
            BLOCK=BLOCK, num_warps=num_warps,
        )

        return scores @ v
