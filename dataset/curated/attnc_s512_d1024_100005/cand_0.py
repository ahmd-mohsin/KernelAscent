import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100005
S, D, DT = 512, 1024, torch.float16


@triton.jit
def _causal_softmax_kernel(
    S_ptr,          # scores, (M, N) fp16, row-major
    O_ptr,          # output probabilities, (M, N) fp16
    N,              # number of columns
    scale,          # 1/sqrt(d)
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    in_bounds = cols < N

    x = tl.load(S_ptr + row * N + cols, mask=in_bounds, other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: columns strictly above the diagonal -> -inf
    x = tl.where(cols <= row, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    tl.store(O_ptr + row * N + cols, p.to(tl.float16), mask=in_bounds)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # ---- fused QKV projection: one big GEMM instead of three ----
        w = getattr(self, "_wqkv", None)
        if w is None or w.device != self.Wq.device or w.dtype != self.Wq.dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._wqkv = w

        qkv = x @ w
        d = x.shape[-1]
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        # ---- scores = q @ k^T (tensor-core GEMM) ----
        scores = torch.matmul(q, k.transpose(-1, -2)).contiguous()

        # ---- fused scale + causal mask + softmax (single Triton kernel, in-place) ----
        if scores.is_cuda:
            M, N = scores.shape[-2], scores.shape[-1]
            BLOCK_N = triton.next_power_of_2(N)
            _causal_softmax_kernel[(M,)](
                scores, scores, N, 1.0 / math.sqrt(d),
                BLOCK_N=BLOCK_N,
                num_warps=8 if BLOCK_N >= 512 else 4,
            )
            a = scores
        else:
            scores = scores / math.sqrt(d)
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)

        # ---- output projection with V ----
        return torch.matmul(a, v)
