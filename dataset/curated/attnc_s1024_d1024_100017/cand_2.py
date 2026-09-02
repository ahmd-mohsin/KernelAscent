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
    S_ptr,          # pointer to scores (in-place)
    N,              # number of columns
    stride_row,     # row stride
    scale,          # 1/sqrt(d)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_bounds = cols < N
    ptrs = S_ptr + row * stride_row + cols

    x = tl.load(ptrs, mask=in_bounds, other=0.0).to(tl.float32)
    x = x * scale
    # causal mask: columns > row get -inf
    causal = cols <= row
    x = tl.where(causal & in_bounds, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(ptrs, y.to(S_ptr.dtype.element_ty), mask=in_bounds)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        # Lazily build a fused QKV weight so all three projections run in one GEMM.
        Wqkv = self.__dict__.get('_Wqkv_cache', None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self.__dict__['_Wqkv_cache'] = Wqkv

        qkv = x @ Wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Attention scores (fp32 accumulate inside cuBLAS tensor-core GEMM)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n_rows, n_cols = scores.shape
        if scores.is_cuda:
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16
            _causal_softmax_kernel[(n_rows,)](
                scores, n_cols, scores.stride(0),
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
