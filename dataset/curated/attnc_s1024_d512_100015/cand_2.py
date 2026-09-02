import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100015
S, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_bounds = cols < N
    s = tl.load(S_ptr + row * N + cols, mask=in_bounds, other=0.0).to(tl.float32)
    s = s * scale
    keep = in_bounds & (cols <= row)
    s = tl.where(keep, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    denom = tl.sum(e, axis=0)
    p = e / denom
    p = tl.where(keep, p, 0.0)
    tl.store(O_ptr + row * N + cols, p.to(O_ptr.dtype.element_ty), mask=in_bounds)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build a fused QKV weight so the three projections run as one GEMM.
        wqkv = self.__dict__.get('_wqkv', None)
        if wqkv is None or wqkv.device != x.device or wqkv.dtype != x.dtype:
            wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous().to(device=x.device, dtype=x.dtype)
            self.__dict__['_wqkv'] = wqkv

        d = x.shape[-1]
        qkv = x @ wqkv
        q, k, v = qkv.split(d, dim=-1)

        # Raw attention logits with a single GEMM (scale fused into softmax kernel).
        scores = torch.matmul(q, k.transpose(-1, -2)).contiguous()

        n = scores.shape[-1]
        rows = scores.shape[0]
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        _causal_softmax_kernel[(rows,)](
            scores, a, n, 1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )

        return a @ v
