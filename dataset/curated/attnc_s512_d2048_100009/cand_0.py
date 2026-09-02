import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100009
S, D, DT = 512, 2048, torch.float16


@triton.jit
def _causal_scale_softmax_kernel(
    x_ptr, out_ptr,
    scale,
    N,
    stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    # one program per row; applies scale, causal mask (cols > row -> -inf),
    # then numerically-stable softmax computed in fp32, output fp16
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    x = tl.where(cols <= row, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        # Fuse the three projections into a single GEMM (dominant cost)
        Wqkv = getattr(self, '_Wqkv_cache', None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv_cache = Wqkv

        qkv = x @ Wqkv
        q, k, v = qkv.split(d, dim=-1)

        # attention scores (unscaled); scaling fused into the softmax kernel
        scores = q @ k.transpose(-1, -2)

        n_rows, n_cols = scores.shape[-2], scores.shape[-1]
        a = torch.empty_like(scores)
        BLOCK_N = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK_N >= 512 else 4
        _causal_scale_softmax_kernel[(n_rows,)](
            scores, a,
            1.0 / math.sqrt(d),
            n_cols,
            scores.stride(-2), a.stride(-2),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

        return a @ v
