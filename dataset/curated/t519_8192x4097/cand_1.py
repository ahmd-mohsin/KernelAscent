import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tld

SEED = 519
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _softmax_scale_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output row (bf16 -> fp32), softmax in fp32 (matches torch acc type)
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tld.exp(x - m)                      # precise expf, matches torch's std::exp
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = e / s

    # torch.softmax on bf16 input returns bf16 -> round here
    p_bf = p.to(tl.bfloat16)

    # x * 1.0722 : bf16 tensor * scalar, computed in fp32 (opmath), rounded to bf16
    y_bf = (p_bf.to(tl.float32) * SCALE).to(tl.bfloat16)

    # _xf = x.float(); rms in fp32
    yf = y_bf.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, 0) / N
    r = tl.math.rsqrt(ms + EPS)

    # (_xf * rsqrt(...)).to(bf16) * weight (bf16*bf16 in fp32 opmath -> bf16)
    o_bf = (yf * r).to(tl.bfloat16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    out = (o_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul on tensor cores via cuBLAS
        h = x @ self.W0
        h = h.contiguous()
        Mr, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_scale_rms_kernel[(Mr,)](
            h, self.rms3_w, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.0722,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
