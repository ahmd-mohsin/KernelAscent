import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100017
S, D, DT = 1024, 1024, torch.float16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    N,
    stride_s, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    # One program per row: fused scale + causal mask + softmax (fp32 math, fp16 io)
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_bounds = cols < N
    s = tl.load(S_ptr + row * stride_s + cols, mask=in_bounds, other=float('-inf')).to(tl.float32)
    s = s * scale
    causal = cols <= row
    s = tl.where(causal, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(causal, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom
    tl.store(O_ptr + row * stride_o + cols, out.to(tl.float16), mask=in_bounds)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        # Lazily build & cache fused QKV weight (single big GEMM instead of three)
        wqkv = getattr(self, '_Wqkv', None)
        if (
            wqkv is None
            or wqkv.device != self.Wq.device
            or wqkv.dtype != self.Wq.dtype
        ):
            wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = wqkv

        # One fused projection GEMM
        qkv = x @ wqkv
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        # Attention scores (cuBLAS fp16 tensor-core GEMM)
        scores = q @ k.transpose(-1, -2)

        n = scores.shape[-1]
        n_rows = scores.numel() // n
        scores_2d = scores.view(n_rows, n)

        # Fused scale + causal mask + softmax, in-place on scores
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _causal_softmax_kernel[(n_rows,)](
            scores_2d, scores_2d,
            n,
            scores_2d.stride(0), scores_2d.stride(0),
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return scores @ v
