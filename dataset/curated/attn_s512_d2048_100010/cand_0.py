import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100010
S, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _scaled_softmax_kernel(
    X_ptr, O_ptr,
    n_cols,
    stride_x, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(O_ptr + row * stride_o + offs, y.to(tl.bfloat16), mask=mask)


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
        d = x.shape[-1]
        Wqkv = self._get_fused_w(x)

        # One fused GEMM for Q, K, V projections
        qkv = x @ Wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Attention scores (fp32 accumulation inside cuBLAS, bf16 output)
        scores = q @ k.transpose(-1, -2)

        # Fused scale + softmax in a single Triton kernel (fp32 math)
        n_rows, n_cols = scores.shape
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _scaled_softmax_kernel[(n_rows,)](
            scores, a,
            n_cols,
            scores.stride(0), a.stride(0),
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
