import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100028
S, D, DT = 2048, 1024, torch.float16


@triton.jit
def _softmax_scale_kernel(
    X_ptr, Y_ptr,
    n_cols,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X_ptr + row * n_cols + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y_ptr + row * n_cols + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None  # lazy fused weight cache

    def forward(self, x):
        d = self.Wq.shape[0]

        # Fuse the three projections into a single GEMM (weights cached & moved with module)
        if (self._Wqkv is None
                or self._Wqkv.device != x.device
                or self._Wqkv.dtype != x.dtype):
            self._Wqkv = torch.cat(
                [self.Wq, self.Wk, self.Wv], dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()

        qkv = x @ self._Wqkv                     # (S, 3D) single big GEMM
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Attention scores (unscaled); scale is fused into the softmax kernel in fp32
        scores = q @ k.transpose(-1, -2)         # (S, S) fp16
        scores = scores.contiguous()

        n_rows, n_cols = scores.shape
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_scale_kernel[(n_rows,)](
            scores, a, n_cols, 1.0 / math.sqrt(d),
            BLOCK=BLOCK, num_warps=num_warps,
        )

        return a @ v
