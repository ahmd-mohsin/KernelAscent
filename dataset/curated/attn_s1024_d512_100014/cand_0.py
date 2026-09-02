import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100014
S, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _scale_softmax_kernel(
    X_ptr, Out_ptr,
    scale,
    N,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Out_ptr + row * stride_o + offs, y.to(Out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._wqkv = None

    @torch.no_grad()
    def forward(self, x):
        # Lazily build fused QKV weight (single GEMM instead of three)
        if (self._wqkv is None
                or self._wqkv.device != x.device
                or self._wqkv.dtype != self.Wq.dtype):
            self._wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous().to(x.device)

        d = self.Wq.shape[1]
        qkv = x @ self._wqkv                       # (S, 3D) one big GEMM
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        # scores = q @ k^T  (scaling fused into softmax kernel)
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores.contiguous()

        n_rows, n_cols = scores.shape
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _scale_softmax_kernel[(n_rows,)](
            scores, a,
            1.0 / math.sqrt(d),
            n_cols,
            scores.stride(0), a.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
