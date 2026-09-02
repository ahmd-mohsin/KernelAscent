import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 364
M, D, DT = 512, 512, torch.float16


@triton.jit
def _gelu_ln_kernel(
    X, G, B, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU, computed in fp32 then rounded to fp16
    # to match F.gelu(fp16) output exactly
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # layernorm stats in fp32 (matches PyTorch fp16 layer_norm accumulation)
    gm = tl.where(mask, g, 0.0)
    mean = tl.sum(gm, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (g - mean) * rstd * gamma + beta
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        M_, N_ = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK >= 1024 else 4
        _gelu_ln_kernel[(M_,)](
            h, self.ln2_g, self.ln2_b, y,
            N_, h.stride(0), y.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
