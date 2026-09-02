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
    # causal mask: positions strictly above the diagonal get -inf
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
        d = self.Wq.shape[0]

        if not x.is_cuda:
            # CPU fallback (reference path)
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        # Fuse the three projections into one big GEMM (cached, no __init__ change)
        Wqkv = getattr(self, '_Wqkv_cache', None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv_cache = Wqkv

        qkv = x @ Wqkv
        q, k, v = qkv.split(d, dim=-1)

        # raw scores (fp16 output, fp32 accumulate inside cuBLAS — matches reference)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n_rows, n_cols = scores.shape
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _causal_softmax_kernel[(n_rows,)](
            scores, scores,
            n_cols, 1.0 / math.sqrt(d),
            scores.stride(0), scores.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return scores @ v
