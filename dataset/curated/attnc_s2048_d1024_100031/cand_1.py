import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100031
S, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    N, stride_s, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ld_mask = offs < N
    x = tl.load(S_ptr + row * stride_s + offs, mask=ld_mask, other=float('-inf'))
    x = x.to(tl.float32) * scale
    # causal mask: keep cols <= row
    x = tl.where(offs <= row, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(O_ptr + row * stride_o + offs, y.to(O_ptr.dtype.element_ty), mask=ld_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Cache fused QKV weight (one big GEMM instead of three)
        Wqkv = getattr(self, '_Wqkv', None)
        if Wqkv is None or Wqkv.device != x.device:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        scores = q @ k.transpose(-1, -2)  # (S, S) bf16
        n = scores.shape[-1]

        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        scale = 1.0 / math.sqrt(q.shape[-1])
        num_warps = 8 if BLOCK >= 2048 else 4
        _causal_softmax_kernel[(scores.shape[0],)](
            scores, a,
            n, scores.stride(0), a.stride(0),
            scale,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
