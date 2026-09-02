import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100035
S, D, DT = 2048, 2048, torch.bfloat16


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

    x = tl.load(S_ptr + row * stride_s + cols, mask=mask,
                other=float('-inf')).to(tl.float32)
    # replicate reference: divide (in bf16 semantics -> fp32 div rounded to bf16)
    x = x / scale
    x = x.to(tl.bfloat16).to(tl.float32)
    # causal mask: columns > row become -inf (matches adding -inf triu mask)
    x = tl.where(cols <= row, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(O_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None  # lazy fused weight cache

    def forward(self, x):
        d = self.Wq.shape[0]

        # Lazily build fused QKV weight (single big GEMM instead of three)
        Wqkv = self._Wqkv
        if (Wqkv is None or Wqkv.device != x.device):
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            if x.is_cuda:
                Wqkv = Wqkv.to(x.device)
            self._Wqkv = Wqkv

        # One fused GEMM for q, k, v
        qkv = x @ Wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Raw attention scores (bf16 GEMM on tensor cores)
        scores = q @ k.transpose(-1, -2)

        n = scores.shape[-1]
        m = scores.shape[-2]

        if scores.is_cuda:
            a = torch.empty_like(scores)
            BLOCK = triton.next_power_of_2(n)
            num_warps = 8 if BLOCK >= 2048 else 4
            _causal_softmax_kernel[(m,)](
                scores, a,
                n, math.sqrt(d),
                scores.stride(0), a.stride(0),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        else:
            scores = scores / math.sqrt(d)
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)

        return a @ v
