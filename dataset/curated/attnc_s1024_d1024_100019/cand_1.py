import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100019
S, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    N, stride_s, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(S_ptr + row * stride_s + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: positions j > i get -inf
    x = tl.where(cols <= row, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(O_ptr + row * stride_o + cols, y.to(O_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Cache fused QKV weight (single big GEMM instead of three)
        Wc = getattr(self, "_Wcat", None)
        if Wc is None or Wc.device != x.device:
            Wc = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wcat = Wc

        d = x.shape[-1]
        qkv = x @ Wc  # (S, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # scores (raw logits, unscaled) via cuBLAS bf16 GEMM
        scores = q @ k.transpose(-1, -2)  # (S, S)

        n = scores.shape[-1]
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        _causal_softmax_kernel[(scores.shape[0],)](
            scores, a,
            n, scores.stride(0), a.stride(0),
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return a @ v
