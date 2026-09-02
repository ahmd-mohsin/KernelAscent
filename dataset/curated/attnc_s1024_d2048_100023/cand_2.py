import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100023
S, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _causal_scale_softmax_kernel(
    S_ptr, O_ptr,
    n_cols, stride_s, stride_o,
    sqrt_d,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    col_mask = cols < n_cols

    x = tl.load(S_ptr + row * stride_s + cols, mask=col_mask, other=float('-inf'))
    # Divide by sqrt(d) in fp32 and round back to bf16 (matches bf16 division
    # semantics of the reference), then upcast for the softmax reduction.
    x = (x.to(tl.float32) / sqrt_d).to(tl.bfloat16).to(tl.float32)

    # Causal mask: positions j > i get -inf (score + (-inf) = -inf in reference).
    x = tl.where(cols <= row, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(O_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=col_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Fuse the three projections into a single large GEMM.
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(x.device, x.dtype).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        # Raw attention scores in bf16 (same as reference q @ k^T).
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n_rows, n_cols = scores.shape
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _causal_scale_softmax_kernel[(n_rows,)](
            scores, a,
            n_cols, scores.stride(0), a.stride(0),
            math.sqrt(q.shape[-1]),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
