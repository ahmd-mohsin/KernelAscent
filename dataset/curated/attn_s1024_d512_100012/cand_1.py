import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100012
S, D, DT = 1024, 512, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    n_cols,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X + row * n_cols + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * n_cols + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Fuse the three projections into a single GEMM (cached fused weight).
        W = getattr(self, '_Wqkv', None)
        if W is None or W.device != x.device or W.dtype != x.dtype:
            W = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous().to(device=x.device, dtype=x.dtype)
            self._Wqkv = W

        d = x.shape[-1]
        qkv = x @ W
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        # Attention scores via cuBLAS (tensor cores).
        scores = q @ k.transpose(-1, -2)  # contiguous output

        if scores.is_cuda:
            n_cols = scores.shape[-1]
            n_rows = scores.numel() // n_cols
            a = torch.empty_like(scores)
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 4
            if BLOCK >= 1024:
                num_warps = 8
            if BLOCK >= 4096:
                num_warps = 16
            _scale_softmax_kernel[(n_rows,)](
                scores, a, n_cols, 1.0 / math.sqrt(d),
                BLOCK=BLOCK, num_warps=num_warps,
            )
        else:
            a = torch.softmax(scores / math.sqrt(d), dim=-1)

        return a @ v
