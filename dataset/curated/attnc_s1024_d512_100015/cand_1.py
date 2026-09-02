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
    N,
    stride_s, stride_o,
    inv_scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(S_ptr + row * stride_s + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # scale (mimic reference: fp32 divide then round to bf16)
    x = x * inv_scale
    x = x.to(tl.bfloat16).to(tl.float32)

    causal = cols <= row
    x = tl.where(causal, x, float('-inf'))

    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(causal & mask, e, 0.0)
    s = tl.sum(e, 0)
    y = e / s

    tl.store(O_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def forward(self, x):
        if self._Wqkv is None or self._Wqkv.device != x.device:
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous().to(x.device)

        d = self.Wq.shape[0]
        n = x.shape[0]

        # Fused QKV projection (single GEMM)
        qkv = x @ self._Wqkv
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        # Raw scores (cuBLAS bf16 GEMM)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        # Fused scale + causal mask + softmax (Triton)
        a = torch.empty_like(scores)
        BLOCK_N = triton.next_power_of_2(n)
        _causal_softmax_kernel[(n,)](
            scores, a,
            n,
            scores.stride(0), a.stride(0),
            1.0 / math.sqrt(d),
            BLOCK_N=BLOCK_N,
            num_warps=8 if BLOCK_N >= 1024 else 4,
        )

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
