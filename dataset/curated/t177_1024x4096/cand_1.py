import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 177
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    N,
    EPS_RMS, EPS_LN,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output row (fp16), relu, upcast to fp32
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + EPS_RMS)
    y16 = (x * rs).to(tl.float16)

    # multiply by rms weight in fp16 (matches reference)
    w16 = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float16)
    z16 = y16 * w16
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)

    # LayerNorm (stats in fp32, affine in fp32, cast to fp16)
    mean = tl.sum(zf, axis=0) / N
    d = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ln16 = (d * inv * g + b).to(tl.float16)

    # ReLU on fp16 values, then softmax in fp32
    r16 = tl.maximum(ln16, 0.0)
    of = r16.to(tl.float32)
    of = tl.where(mask, of, float("-inf"))
    mx = tl.max(of, axis=0)
    e = tl.exp(of - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_post_kernel[(Mrows,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, out,
            N, 1e-6, 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
