import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 47
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_bias_ln_rms_kernel(
    X_ptr, B1_ptr, G_ptr, B_ptr, W_ptr, Out_ptr,
    N,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output row (bf16) and bias
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b1  (bf16 add: fp32 compute, round to bf16)
    xb = (x + b1).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch internal fp32 accumulation), output bf16
    mean = tl.sum(tl.where(mask, xb, 0.0), axis=0) / N
    d = tl.where(mask, xb - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (d * inv * g + b).to(tl.bfloat16).to(tl.float32)

    # x = x * 1.0003 (bf16 result)
    z = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32, cast to bf16, multiply by weight in bf16
    ms = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / N
    r = z * (1.0 / tl.sqrt(ms + EPS_RMS))
    r_bf = r.to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (r_bf * w).to(tl.bfloat16)

    tl.store(Out_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores (bf16)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_bias_ln_rms_kernel[(Mrows,)](
            h, self.b1, self.ln2_g, self.ln2_b, self.rms4_w, out,
            N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            SCALE=1.0003,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
