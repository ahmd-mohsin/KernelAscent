import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100020
S, D, DT = 1024, 2048, torch.float16


@triton.jit
def _scaled_softmax_kernel(
    X, Y,
    scale,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + offs, y.to(Y.dtype.element_ty), mask=mask)


def _scaled_softmax(scores: torch.Tensor, scale: float) -> torch.Tensor:
    M, N = scores.shape
    out = torch.empty_like(scores)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK >= 2048:
        num_warps = 8
    if BLOCK >= 8192:
        num_warps = 16
    _scaled_softmax_kernel[(M,)](
        scores, out, scale, N,
        scores.stride(0), out.stride(0),
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    @torch.no_grad()
    def forward(self, x):
        # Lazily build fused QKV weight (one big GEMM instead of three)
        if (self._Wqkv is None
                or self._Wqkv.device != x.device
                or self._Wqkv.dtype != x.dtype):
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(
                device=x.device, dtype=x.dtype
            ).contiguous()

        d = self.Wq.shape[1]
        qkv = x @ self._Wqkv                     # (S, 3D) single GEMM
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        scores = q @ k.transpose(-1, -2)         # (S, S)
        scale = 1.0 / math.sqrt(d)

        if scores.is_cuda:
            a = _scaled_softmax(scores, scale)   # fused scale + softmax (fp32 accum)
        else:
            a = torch.softmax(scores * scale, dim=-1)

        return a @ v
