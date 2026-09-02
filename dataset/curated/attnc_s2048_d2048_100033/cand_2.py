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
    S_ptr,
    stride_row,
    n_cols,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    # causal: only columns <= row are valid
    valid = (offs < n_cols) & (offs <= row)
    ptrs = S_ptr + row * stride_row + offs

    x = tl.load(ptrs, mask=valid, other=float('-inf')).to(tl.float32)
    x = x * scale
    x = tl.where(valid, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(valid, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(ptrs, y.to(S_ptr.dtype.element_ty), mask=offs < n_cols)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = x.shape[-1]
        n = x.shape[0]

        # Lazily build fused QKV weight (single big GEMM instead of three)
        Wqkv = getattr(self, '_Wqkv', None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        qkv = x @ Wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # raw scores (unscaled); scaling + causal mask + softmax fused in Triton
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        if scores.is_cuda:
            BLOCK = triton.next_power_of_2(n)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16
            _causal_softmax_kernel[(n,)](
                scores,
                scores.stride(0),
                n,
                1.0 / math.sqrt(d),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
            a = scores
        else:
            s = scores.float() / math.sqrt(d)
            s = s + torch.triu(torch.full_like(s, float('-inf')), diagonal=1)
            a = torch.softmax(s, dim=-1).to(scores.dtype)

        return a @ v
