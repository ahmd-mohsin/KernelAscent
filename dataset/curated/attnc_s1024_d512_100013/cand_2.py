import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100013
S, D, DT = 1024, 512, torch.float16


@triton.jit
def _causal_scale_softmax_kernel(
    S_ptr, O_ptr,
    n_cols, scale,
    stride_s, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(S_ptr + row * stride_s + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: cols > row -> -inf
    x = tl.where(cols <= row, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(O_ptr + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        # Cache fused QKV weight (single GEMM instead of three)
        wqkv = getattr(self, '_Wqkv', None)
        if wqkv is None or wqkv.device != x.device or wqkv.dtype != self.Wq.dtype:
            wqkv = torch.cat((self.Wq, self.Wk, self.Wv), dim=1).contiguous()
            self._Wqkv = wqkv

        orig_shape = x.shape
        x2 = x.reshape(-1, d)
        n = x2.shape[0]

        qkv = x2 @ wqkv  # (n, 3d) in one GEMM
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # scores = q @ k^T (cuBLAS tensor-core GEMM)
        scores = q @ k.transpose(-1, -2)  # (n, n), contiguous

        # Fused: scale + causal mask + softmax (in-place, single kernel)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _causal_scale_softmax_kernel[(n,)](
            scores, scores,
            n, 1.0 / math.sqrt(d),
            scores.stride(0), scores.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        out = scores @ v
        return out.reshape(orig_shape)
