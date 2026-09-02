import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100013
S, D, DT = 1024, 512, torch.float16


@triton.jit
def _causal_softmax_kernel(
    S_ptr,
    stride_row,
    n_cols,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    col_mask = cols < n_cols
    ptrs = S_ptr + row * stride_row + cols

    x = tl.load(ptrs, mask=col_mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: only positions <= row are valid
    x = tl.where(cols <= row, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(ptrs, y.to(S_ptr.dtype.element_ty), mask=col_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = x.shape[-1]
        # Cache the fused QKV weight (single big GEMM instead of three)
        wqkv = getattr(self, '_Wqkv', None)
        if (
            wqkv is None
            or wqkv.device != self.Wq.device
            or wqkv.dtype != self.Wq.dtype
        ):
            wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = wqkv

        qkv = x @ wqkv
        q, k, v = qkv.split(d, dim=-1)

        # scores = q @ k^T (tensor-core GEMM); scale + causal mask + softmax fused in Triton
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n = scores.shape[-1]
        if scores.is_cuda:
            BLOCK = triton.next_power_of_2(n)
            num_warps = 4
            if BLOCK >= 1024:
                num_warps = 8
            if BLOCK >= 4096:
                num_warps = 16
            _causal_softmax_kernel[(scores.shape[0],)](
                scores,
                scores.stride(0),
                n,
                1.0 / math.sqrt(d),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
            a = scores
        else:
            scores = scores / math.sqrt(d)
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)

        return a @ v
