import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100026
S, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _softmax_kernel(S_ptr, O_ptr, n_cols, sqrt_d, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    s = tl.load(S_ptr + row * n_cols + offs, mask=mask, other=float('-inf')).to(tl.float32)
    # replicate: (scores / sqrt(d)) rounded to bf16, then softmax in fp32
    s = (s / sqrt_d).to(tl.bfloat16).to(tl.float32)
    m = tl.max(s, 0)
    e = tl.exp(s - m)
    denom = tl.sum(e, 0)
    out = e / denom
    tl.store(O_ptr + row * n_cols + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        if not x.is_cuda:
            # CPU fallback (reference path)
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v

        # Cache fused QKV weight (single wide GEMM instead of three)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        qkv = x @ Wqkv
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()
        n_rows, n_cols = scores.shape[-2], scores.shape[-1]
        scores_2d = scores.view(-1, n_cols)
        a = torch.empty_like(scores_2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_kernel[(scores_2d.shape[0],)](
            scores_2d, a, n_cols, math.sqrt(q.shape[-1]),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        a = a.view_as(scores)
        return a @ v
