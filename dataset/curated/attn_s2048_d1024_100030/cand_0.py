import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100030
S, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _scaled_softmax_kernel(
    X, Y, N, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row.to(tl.int64) * N + offs
    x = tl.load(ptr, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    optr = Y + row.to(tl.int64) * N + offs
    tl.store(optr, y.to(Y.dtype.element_ty), mask=mask)


def _scaled_softmax(scores: torch.Tensor, scale: float) -> torch.Tensor:
    M, N = scores.shape
    out = torch.empty_like(scores)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK >= 2048:
        num_warps = 8
    if BLOCK >= 8192:
        num_warps = 16
    _scaled_softmax_kernel[(M,)](scores, out, N, scale, BLOCK=BLOCK, num_warps=num_warps)
    return out


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._wqkv_cache = None

    def forward(self, x):
        d = self.Wq.shape[0]
        # Fuse the three projection GEMMs into a single GEMM (cache the concat weight)
        if (self._wqkv_cache is None
                or self._wqkv_cache.device != x.device
                or self._wqkv_cache.dtype != self.Wq.dtype):
            self._wqkv_cache = torch.cat(
                (self.Wq, self.Wk, self.Wv), dim=1
            ).to(x.device).contiguous()
        Wqkv = self._wqkv_cache

        qkv = x @ Wqkv                       # (S, 3D) single big GEMM
        q, k, v = qkv.split(d, dim=-1)

        scores = q @ k.transpose(-1, -2)     # (S, S) raw scores in bf16

        if scores.is_cuda:
            # fused scale + softmax in one Triton kernel (fp32 accumulation)
            a = _scaled_softmax(scores.contiguous(), 1.0 / math.sqrt(d))
        else:
            a = torch.softmax(scores / math.sqrt(d), dim=-1)

        return a @ v
