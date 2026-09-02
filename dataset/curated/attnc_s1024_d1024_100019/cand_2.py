import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100019
S, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    n_cols, stride_s, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    s = tl.load(S_ptr + row * stride_s + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # match reference: (bf16 scores) / sqrt(d) rounds to bf16 before softmax
    s = (s * scale).to(tl.bfloat16).to(tl.float32)
    # causal mask: cols > row -> -inf
    s = tl.where(cols <= row, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    d = tl.sum(e, axis=0)
    p = e / d
    tl.store(O_ptr + row * stride_o + cols, p.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference implementation
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        d = self.Wq.shape[0]

        # QKV projections (kept as separate GEMMs for bitwise-identical q,k,v)
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv

        # raw attention scores (bf16 GEMM, same as reference)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n = scores.shape[-1]
        m = scores.shape[-2]
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        # fused scale + causal mask + softmax (in-place on scores buffer)
        _causal_softmax_kernel[(m,)](
            scores, scores,
            n, scores.stride(-2), scores.stride(-2),
            1.0 / math.sqrt(q.shape[-1]),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return scores @ v
