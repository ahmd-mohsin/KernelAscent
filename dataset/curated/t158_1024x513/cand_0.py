import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 158
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_softmax_kernel(
    X, B1, B3, OUT,
    n_cols,
    stride_xm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)

    # x = x * 1.0026 (compute in fp32, round to bf16 like PyTorch)
    t = (x.to(tl.float32) * 1.0026).to(tl.bfloat16)
    # x = x + b1 (fp32 opmath, round to bf16)
    t = (t.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch bf16 softmax accumulation), output bf16
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    mx = tl.max(tf, axis=0)
    e = tl.exp(tf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # x = x + b3, then x = x * 1.0435 (fp32 opmath, round bf16 each step)
    o = (sm.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)
    o = (o.to(tl.float32) * 1.0435).to(tl.bfloat16)

    tl.store(OUT + row * stride_om + cols, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x * 1.0026
            y = y + self.b1
            y = torch.softmax(y, dim=-1)
            y = y + self.b3
            y = y * 1.0435
            return y

        x = x.contiguous()
        rows, n_cols = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        x2 = x.view(-1, n_cols)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_softmax_kernel[(x2.shape[0],)](
            x2, self.b1, self.b3, out,
            n_cols,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return out.view(x.shape)
