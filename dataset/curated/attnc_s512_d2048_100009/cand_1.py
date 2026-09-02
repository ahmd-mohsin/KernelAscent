import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100009
S, D, DT = 512, 2048, torch.float16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    n_cols,
    stride_s, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(S_ptr + row * stride_s + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32) * scale
    # causal mask: positions with col > row get -inf
    x = tl.where(cols <= row, x, float('-inf'))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(O_ptr + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_w(self):
        w = getattr(self, '_Wqkv', None)
        if w is None or w.device != self.Wq.device or w.dtype != self.Wq.dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        d = x.shape[-1]
        Wqkv = self._get_fused_w()

        # Single fused GEMM for Q, K, V projections
        qkv = x @ Wqkv
        q, k, v = qkv.split(d, dim=-1)

        # Raw attention scores (unscaled); scaling fused into softmax kernel
        scores = torch.matmul(q, k.transpose(-1, -2))

        n = scores.shape[-1]
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 512 else 4
        _causal_softmax_kernel[(scores.shape[0],)](
            scores, a,
            n,
            scores.stride(0), a.stride(0),
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return torch.matmul(a, v)
