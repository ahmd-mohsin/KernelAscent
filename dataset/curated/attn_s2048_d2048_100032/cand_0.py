import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100032
S, D, DT = 2048, 2048, torch.float16


@triton.jit
def _scaled_softmax_kernel(ptr, N, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    p = ptr + row.to(tl.int64) * N + offs
    x = tl.load(p, mask=mask, other=float('-inf')).to(tl.float32)
    # mimic reference: scores = (q @ k^T) / sqrt(d) stored in fp16, then softmax
    x = (x * scale).to(tl.float16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    y = e / s
    tl.store(p, y.to(tl.float16), mask=mask)


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

        if not x.is_cuda:
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v

        w = self._get_fused_w(x)

        # single fused QKV projection (one big GEMM on tensor cores)
        qkv = x @ w
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        # raw scores (unscaled); scale is fused into the softmax kernel
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n = scores.shape[-1]
        rows = scores.numel() // n
        BLOCK = triton.next_power_of_2(n)
        scale = 1.0 / math.sqrt(q.shape[-1])
        num_warps = 8 if BLOCK >= 2048 else 4
        _scaled_softmax_kernel[(rows,)](scores, n, scale, BLOCK=BLOCK, num_warps=num_warps)

        return scores @ v
