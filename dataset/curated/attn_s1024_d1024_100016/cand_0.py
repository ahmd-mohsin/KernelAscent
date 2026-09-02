import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100016
S, D, DT = 1024, 1024, torch.float16


@triton.jit
def _softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + offs, y.to(Y.dtype.element_ty), mask=mask)


def _triton_softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK >= 2048:
        num_warps = 8
    if BLOCK >= 8192:
        num_warps = 16
    _softmax_kernel[(n_rows,)](
        x, y, n_cols, x.stride(0), y.stride(0),
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[1]
        scale = 1.0 / math.sqrt(d)

        # Cache fused QKV weight with the softmax scale folded into Wq.
        # scale = 1/32 is an exact power of two, so folding it into the
        # weights is bit-exact with scaling the scores afterwards.
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat(
                [self.Wq * scale, self.Wk, self.Wv], dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        # One fused GEMM for Q, K, V projections.
        qkv = x @ Wqkv
        q, k, v = qkv.chunk(3, dim=-1)

        # Attention scores (scale already folded into q via Wq).
        scores = q @ k.transpose(-1, -2)

        if scores.is_cuda:
            a = _triton_softmax(scores)
        else:
            a = torch.softmax(scores, dim=-1)

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
