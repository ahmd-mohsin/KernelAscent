import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100011
S, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    L,                      # sequence length (mask period)
    N,                      # row length (== L here)
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    row_in_seq = row % L

    cols = tl.arange(0, BLOCK)
    cmask = cols < N

    x = tl.load(S_ptr + row * N + cols, mask=cmask, other=float('-inf')).to(tl.float32)

    # emulate the reference exactly: (scores / sqrt(d)) rounded to the score dtype,
    # then softmax computed in fp32 (matching torch's upcasting softmax)
    x = x * scale
    x = x.to(S_ptr.dtype.element_ty).to(tl.float32)

    valid = cols <= row_in_seq
    x = tl.where(valid & cmask, x, float('-inf'))

    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(valid & cmask, e, 0.0)
    s = tl.sum(e, 0)
    y = e / s

    tl.store(O_ptr + row * N + cols, y.to(O_ptr.dtype.element_ty), mask=cmask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_w(self, x):
        w = getattr(self, '_Wqkv', None)
        if w is None or w.device != x.device or w.dtype != self.Wq.dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(x.device).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = self.Wq.shape[0]

        if not x.is_cuda:
            # CPU fallback: reference path
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        Wqkv = self._get_fused_w(x)

        # Single fused projection GEMM for Q, K, V
        qkv = x @ Wqkv
        q, k, v = qkv.split(d, dim=-1)

        # Attention scores (cuBLAS handles the strided views)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        L = scores.shape[-1]
        M = scores.numel() // L
        a = torch.empty_like(scores)

        BLOCK = triton.next_power_of_2(L)
        num_warps = 4 if BLOCK <= 1024 else 8
        _causal_softmax_kernel[(M,)](
            scores, a,
            scores.shape[-2], L,
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
