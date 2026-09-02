import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100032
S, D, DT = 2048, 2048, torch.float16


@triton.jit
def _scaled_softmax_kernel(
    X, Out,
    N,
    stride_xm, stride_om,
    inv_scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # replicate fp16 division rounding of the reference (scores / sqrt(d) in fp16)
    x = x * inv_scale  # fp16 arithmetic, matches fp16 elementwise div rounding closely
    xf = x.to(tl.float32)
    row_max = tl.max(xf, axis=0)
    xf = xf - row_max
    num = tl.exp(xf)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(Out + row * stride_om + cols, y.to(Out.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_w(self):
        w = getattr(self, "_Wqkv", None)
        if w is None or w.device != self.Wq.device or w.dtype != self.Wq.dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = w
        return w

    @torch.no_grad()
    def forward(self, x):
        d = self.Wq.shape[1]

        if not x.is_cuda:
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v

        Wqkv = self._get_fused_w()

        # single fused GEMM for all three projections
        qkv = x @ Wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        scores = q @ k.transpose(-1, -2)

        M, N = scores.shape
        a = torch.empty_like(scores)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _scaled_softmax_kernel[(M,)](
            scores, a,
            N,
            scores.stride(0), a.stride(0),
            1.0 / math.sqrt(d),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

        return a @ v
