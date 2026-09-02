import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100023
S, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    N, inv_scale,
    stride_s, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    in_bounds = offs < N

    x = tl.load(S_ptr + row * stride_s + offs, mask=in_bounds,
                other=float('-inf')).to(tl.float32)
    # match reference: scores / sqrt(d)
    x = x / inv_scale
    # causal mask: keep j <= row
    causal = offs <= row
    x = tl.where(causal & in_bounds, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(causal & in_bounds, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(O_ptr + row * stride_o + offs, y.to(tl.bfloat16), mask=in_bounds)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def forward(self, x):
        # Lazily build fused QKV weight (single big GEMM instead of three)
        if (self._Wqkv is None
                or self._Wqkv.device != x.device
                or self._Wqkv.dtype != self.Wq.dtype):
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()

        d = self.Wq.shape[0]
        n = x.shape[0]

        qkv = x @ self._Wqkv                       # (S, 3D) one GEMM
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        scores = q @ k.transpose(-1, -2)           # (S, S) bf16 GEMM
        scores = scores.contiguous()

        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        _causal_softmax_kernel[(n,)](
            scores, a,
            n, math.sqrt(d),
            scores.stride(0), a.stride(0),
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )

        return a @ v
