import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100021
S, D, DT = 1024, 2048, torch.float16


@triton.jit
def _causal_softmax_kernel(
    S_ptr,
    stride_row,
    N,
    inv_scale,  # 1/sqrt(D)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    ptrs = S_ptr + row * stride_row + cols

    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    # emulate reference: fp16 division by sqrt(D), then softmax in fp32
    x16 = (x * inv_scale).to(tl.float16)
    xf = x16.to(tl.float32)

    valid = mask & (cols <= row)
    xf = tl.where(valid, xf, float('-inf'))

    m = tl.max(xf, 0)
    e = tl.exp(xf - m)
    e = tl.where(valid, e, 0.0)
    s = tl.sum(e, 0)
    y = (e / s).to(tl.float16)

    tl.store(ptrs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_w(self, x):
        w = getattr(self, "_Wqkv", None)
        if (
            w is None
            or w.device != x.device
            or w.dtype != x.dtype
        ):
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        if not x.is_cuda:
            # fallback (CPU) - reference path
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        d = x.shape[-1]
        w = self._get_fused_w(x)

        # fused QKV projection: one big GEMM
        qkv = x @ w  # (S, 3D)
        q, k, v = qkv.split(d, dim=-1)

        # attention scores via cuBLAS
        scores = q @ k.transpose(-1, -2)  # fp16, (S, S)
        scores = scores.contiguous()

        n = scores.shape[-1]
        n_rows = scores.shape[0]
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4

        _causal_softmax_kernel[(n_rows,)](
            scores,
            scores.stride(0),
            n,
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return scores @ v
