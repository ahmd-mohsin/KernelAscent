import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100000
S, D, DT = 512, 512, torch.float16


@triton.jit
def _softmax_scale_kernel(
    x_ptr, out_ptr,
    n_cols,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=float('-inf'))
    x = x.to(tl.float32) * scale
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    denom = tl.sum(e, axis=0)
    y = e / denom
    tl.store(out_ptr + row * n_cols + offs,
             y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build a fused QKV weight so all three projections run as one GEMM.
        Wqkv = getattr(self, "_Wqkv", None)
        if (Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype
                or getattr(self, "_Wqkv_src", None) is not self.Wq):
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(
                device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv
            self._Wqkv_src = self.Wq

        d = self.Wq.shape[1]
        qkv = x @ Wqkv  # (S, 3D) single GEMM instead of three
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        if not x.is_cuda:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        # Raw attention scores (unscaled); scaling fused into the softmax kernel.
        scores = torch.matmul(q, k.transpose(-1, -2)).contiguous()

        n_rows, n_cols = scores.shape
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _softmax_scale_kernel[(n_rows,)](
            scores, a, n_cols, 1.0 / math.sqrt(d),
            BLOCK=BLOCK, num_warps=num_warps,
        )

        return a @ v
