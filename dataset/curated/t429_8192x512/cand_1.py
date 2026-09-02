import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 429
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _rms_softmax_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load row (bf16 -> fp32)
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32 (matches _xf.pow(2).mean(-1) then rsqrt)
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + EPS)

    # cast normalized value to bf16, multiply by weight in bf16 (matches
    # (.to(x.dtype)) * rms1_w which is a bf16*bf16 elementwise op)
    xn = (x * r).to(tl.bfloat16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    v = xn * w  # bf16 result

    # softmax with fp32 accumulation (matches PyTorch bf16 softmax semantics)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    mx = tl.max(vf, axis=0)
    e = tl.exp(vf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (bf16 tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape

        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4

        _rms_softmax_kernel[(m,)](
            h, self.rms1_w, y,
            n, h.stride(0), y.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
