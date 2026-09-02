import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100021
S, D, DT = 1024, 2048, torch.float16


@triton.jit
def _causal_scale_softmax_kernel(
    S_ptr, O_ptr,
    n_cols, scale,
    stride_s, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    causal_mask = cols <= row
    load_mask = causal_mask & (cols < n_cols)

    s = tl.load(S_ptr + row * stride_s + cols, mask=load_mask,
                other=float('-inf')).to(tl.float32)
    s = s * scale

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(load_mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(O_ptr + row * stride_o + cols,
             out.to(O_ptr.dtype.element_ty),
             mask=cols < n_cols)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Cache fused QKV weight (single GEMM instead of three)
        Wqkv = getattr(self, '_Wqkv', None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(
                device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        d = x.shape[-1]
        qkv = x @ Wqkv
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        # Raw attention scores (scale is fused into the softmax kernel)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n = scores.shape[-1]
        a = torch.empty_like(scores)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 1024:
            num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16

        _causal_scale_softmax_kernel[(scores.shape[0],)](
            scores, a,
            n, 1.0 / math.sqrt(d),
            scores.stride(0), a.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
