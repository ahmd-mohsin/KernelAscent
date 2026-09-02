import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100015
S, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    n_cols, sqrt_d,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(S_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    # match reference: division to bf16 precision before softmax
    x = (x / sqrt_d).to(tl.bfloat16).to(tl.float32)
    # causal mask: cols > row -> -inf
    x = tl.where(cols <= row, x, float('-inf'))

    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    y = e / s

    tl.store(O_ptr + row * stride_row + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # reference fallback on CPU
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        d = x.shape[-1]

        # fused QKV projection (cache concatenated weight)
        W = getattr(self, "_Wqkv", None)
        if W is None or W.device != x.device or W.dtype != x.dtype:
            W = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(x.device, x.dtype).contiguous()
            self._Wqkv = W

        qkv = x @ W
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # raw scores (bf16 tensor-core GEMM, fp32 accumulate — same as reference)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n = scores.shape[0]
        n_cols = scores.shape[1]
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 1024 else 4

        _causal_softmax_kernel[(n,)](
            scores, scores,
            n_cols, math.sqrt(d),
            scores.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return scores @ v
