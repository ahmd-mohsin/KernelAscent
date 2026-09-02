import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100018
S, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _scaled_softmax_kernel(
    X_ptr, Y_ptr,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y_ptr + row * N + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_wqkv(self):
        w = getattr(self, "_wqkv_cache", None)
        if (
            w is None
            or w.device != self.Wq.device
            or w.dtype != self.Wq.dtype
        ):
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._wqkv_cache = w
        return w

    @torch.no_grad()
    def forward(self, x):
        d = self.Wq.shape[0]

        if not x.is_cuda:
            # CPU fallback: reference path
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v

        # Fused QKV projection: single large GEMM instead of three
        wqkv = self._get_wqkv()
        qkv = x @ wqkv  # (S, 3D)
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Attention scores (cuBLAS bf16 GEMM)
        scores = q @ k.transpose(-1, -2)  # (S, S)
        scores = scores.contiguous()

        # Fused scale + softmax in a single Triton kernel (one pass over memory)
        n_rows, n_cols = scores.shape
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _scaled_softmax_kernel[(n_rows,)](
            scores, a,
            n_cols,
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
