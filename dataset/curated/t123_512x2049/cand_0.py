import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 123
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_softmax_rms_relu(X, W, Y, n_cols, stride_x, stride_y,
                            eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch's bf16 softmax path)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    p = e / s

    # cast to bf16 then back to fp32 (matches x.float() after bf16 softmax output)
    pb = p.to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, pb * pb, 0.0), axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + eps)

    # normalized value cast to bf16 (matches .to(x.dtype)), then multiply by weight
    yn = (pb * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = yn * w

    # relu(relu(x)) == relu(x)
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_rms_relu[(m,)](
            h, self.rms2_w, out,
            n, h.stride(0), out.stride(0),
            1e-6, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
