import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100022
S, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _scaled_softmax_kernel(
    X, Y,
    stride_x, stride_y,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    # replicate: (scores / sqrt(d)) in bf16, then softmax in fp32
    x = (x * scale).to(tl.bfloat16).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_w(self, x):
        w = getattr(self, "_Wqkv", None)
        if w is None or w.device != x.device or w.dtype != x.dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = self.Wq.shape[0]
        scale = 1.0 / math.sqrt(d)

        if not x.is_cuda:
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) * scale
            a = torch.softmax(scores, dim=-1)
            return a @ v

        # Fused QKV projection: one big GEMM instead of three
        w = self._get_fused_w(x)
        qkv = x @ w
        q, k, v = qkv.split(d, dim=-1)

        # Attention scores (bf16 GEMM, fp32 accumulate via cuBLAS)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n_rows, n_cols = scores.shape[-2], scores.shape[-1]
        scores_2d = scores.view(-1, n_cols)
        a = torch.empty_like(scores_2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _scaled_softmax_kernel[(scores_2d.shape[0],)](
            scores_2d, a,
            scores_2d.stride(0), a.stride(0),
            n_cols, scale,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        a = a.view_as(scores)
        return a @ v
