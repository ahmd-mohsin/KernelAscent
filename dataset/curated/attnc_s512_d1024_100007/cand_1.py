import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100007
S, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr,          # scores, bf16, (n, n) row-major
    O_ptr,          # output probs, bf16, (n, n)
    n,              # sequence length
    scale,          # 1/sqrt(d)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n

    s = tl.load(S_ptr + row * n + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # match reference: division performed in bf16 precision before softmax
    s = (s * scale).to(tl.bfloat16).to(tl.float32)
    # causal mask: positions j > i get -inf
    s = tl.where(cols <= row, s, float('-inf'))

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    denom = tl.sum(e, axis=0)
    p = e / denom

    tl.store(O_ptr + row * n + cols, p.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None  # lazily-built fused projection weight (not a parameter)

    def forward(self, x):
        # Lazily build / refresh fused QKV weight on the right device
        if (self._Wqkv is None
                or self._Wqkv.device != self.Wq.device
                or self._Wqkv.dtype != self.Wq.dtype):
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()

        d = x.shape[-1]
        n = x.shape[-2]

        # single fused GEMM for Q, K, V projections
        qkv = x @ self._Wqkv
        q, k, v = qkv.split(d, dim=-1)

        # raw attention scores in bf16 (tensor-core GEMM)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        # fused scale + causal mask + softmax (in-place, one program per row)
        BLOCK = triton.next_power_of_2(n)
        _causal_softmax_kernel[(n,)](
            scores, scores, n, 1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 512 else 4,
        )

        return scores @ v
