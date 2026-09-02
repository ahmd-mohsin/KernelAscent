import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100000
S, D, DT = 512, 512, torch.float16


@triton.jit
def _scaled_softmax_kernel(
    x_ptr, out_ptr,
    scale,
    n_cols,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=float('-inf'))
    x = x.to(tl.float32) * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(out_ptr + row * stride_o + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def forward(self, x):
        # Lazily build fused QKV weight (single GEMM instead of three)
        if (self._Wqkv is None or self._Wqkv.device != x.device
                or self._Wqkv.dtype != self.Wq.dtype):
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()

        d = self.Wq.shape[1]
        qkv = x @ self._Wqkv                      # one fused GEMM
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        scores = q @ k.transpose(-1, -2)          # raw scores (fp16 GEMM, tensor cores)

        n_rows, n_cols = scores.shape
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _scaled_softmax_kernel[(n_rows,)](
            scores, a,
            1.0 / math.sqrt(q.shape[-1]),
            n_cols,
            scores.stride(0), a.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
