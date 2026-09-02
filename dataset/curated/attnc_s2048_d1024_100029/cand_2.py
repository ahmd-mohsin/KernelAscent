import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100029
S, D, DT = 2048, 1024, torch.float16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    N, scale,
    stride_s, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_bounds = cols < N
    causal = cols <= row

    s = tl.load(S_ptr + row * stride_s + cols, mask=in_bounds, other=0.0).to(tl.float32)
    s = s * scale
    s = tl.where(causal & in_bounds, s, float('-inf'))

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    denom = tl.sum(e, axis=0)
    p = e / denom

    tl.store(O_ptr + row * stride_o + cols, p.to(tl.float16), mask=in_bounds)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def forward(self, x):
        # Fuse the three projection GEMMs into a single GEMM (cached concat).
        if self._Wqkv is None or self._Wqkv.device != x.device:
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()

        n, d = x.shape[-2], x.shape[-1]
        qkv = x @ self._Wqkv
        q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]

        # Raw attention scores (unscaled) via cuBLAS GEMM.
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        # Fused (scale + causal mask + softmax) in one Triton kernel, fp32 math.
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _causal_softmax_kernel[(n,)](
            scores, a,
            n, 1.0 / math.sqrt(d),
            scores.stride(-2), a.stride(-2),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
