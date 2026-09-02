import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100033
S, D, DT = 2048, 2048, torch.float16


@triton.jit
def _causal_softmax_kernel(
    S_ptr,          # [M, N] fp16 scores (in/out)
    N,              # number of columns
    scale,          # 1/sqrt(d)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    col_mask = cols < N
    ptr = S_ptr + row * N + cols
    x = tl.load(ptr, mask=col_mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: positions strictly above diagonal -> -inf
    x = tl.where(cols <= row, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(ptr, y.to(tl.float16), mask=col_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference implementation)
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        d = self.Wq.shape[0]

        # Fuse the three projections into a single GEMM (cache the concatenated weight)
        Wqkv = getattr(self, '_Wqkv_cache', None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(x.device, x.dtype).contiguous()
            self._Wqkv_cache = Wqkv

        qkv = x @ Wqkv
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        # scores = q @ k^T (fp16 GEMM on tensor cores)
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores.contiguous()

        M, N = scores.shape
        scale = 1.0 / math.sqrt(d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _causal_softmax_kernel[(M,)](
            scores, N, scale,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return torch.matmul(scores, v)
