import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

SEED = 100028
S, D, DT = 2048, 1024, torch.float16


@triton.jit
def _softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    y = e / s
    tl.store(Y + row * stride_y + offs, y.to(Y.dtype.element_ty), mask=mask)


def _triton_softmax_(x):
    # in-place row softmax
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK >= 2048:
        num_warps = 8
    if BLOCK >= 8192:
        num_warps = 16
    _softmax_kernel[(M,)](x, x, N, x.stride(0), x.stride(0),
                          BLOCK=BLOCK, num_warps=num_warps)
    return x


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_weight(self, device, dtype):
        W = getattr(self, "_Wqkv", None)
        if W is None or W.device != device or W.dtype != dtype:
            d = self.Wq.shape[-1]
            # 1/sqrt(1024) = 2^-5: exact power-of-two scale, folding it into Wq
            # is bit-exact vs. dividing the scores afterwards.
            scale = 1.0 / math.sqrt(d)
            Wq_scaled = (self.Wq.to(device=device, dtype=dtype) * scale)
            W = torch.cat(
                [Wq_scaled,
                 self.Wk.to(device=device, dtype=dtype),
                 self.Wv.to(device=device, dtype=dtype)],
                dim=1,
            ).contiguous()
            self._Wqkv = W
        return W

    def forward(self, x):
        d = self.Wq.shape[-1]
        W = self._get_fused_weight(x.device, x.dtype)

        # Single fused GEMM for Q, K, V projections
        qkv = x @ W
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Scale already folded into Wq (exact power-of-two)
        scores = q @ k.transpose(-1, -2)

        if scores.is_cuda:
            a = _triton_softmax_(scores)
        else:
            a = torch.softmax(scores, dim=-1)

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
