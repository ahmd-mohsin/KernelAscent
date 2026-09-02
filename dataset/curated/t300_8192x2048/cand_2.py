import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_scale_relu_ln_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul output (fp16) and apply scale (computed in fp32, rounded to fp16
    # to match PyTorch half-tensor * python-scalar semantics)
    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    v = (x * 1.0809).to(tl.float16)
    # relu (exact in fp16)
    v = tl.where(v > 0, v, 0.0).to(tl.float16)

    # layer_norm in fp32 (matches PyTorch internal fp32 accumulation), eps=1e-5
    vf = v.to(tl.float32)
    mean = tl.sum(tl.where(mask, vf, 0.0), axis=0) / N
    d = tl.where(mask, vf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * inv * g + b).to(tl.float16)  # LN output rounded to fp16

    # scale by 1.4484 (fp32 compute, fp16 round)
    z = (y.to(tl.float32) * 1.4484).to(tl.float16)

    # RMS norm in fp32 exactly like reference: _xf = z.float()
    zf = z.to(tl.float32)
    ms = tl.sum(tl.where(mask, zf * zf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    o = (zf * r).to(tl.float16)  # .to(dtype) rounding in reference

    # multiply by rms weight (half * half -> fp32 opmath -> fp16 round)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (o.to(tl.float32) * w).to(tl.float16)

    tl.store(Out_ptr + row * stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_scale_relu_ln_rms_kernel[(m,)](
            h, self.ln3_g, self.ln3_b, self.rms5_w, out,
            n, h.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
