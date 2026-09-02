import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 620
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, B1_ptr, G_ptr, B_ptr, B4_ptr, W_ptr, OUT_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    # load matmul output row (fp16)
    x = tl.load(X_ptr + row * N + cols)
    b1 = tl.load(B1_ptr + cols)

    # bias add in fp16 (matches half add semantics)
    h = x + b1

    # GELU (exact, erf) computed in fp32, cast back to fp16 (matches PyTorch opmath)
    hf = h.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865475))
    gh = g.to(tl.float16)

    # LayerNorm computed in fp32, output fp16
    gf = gh.to(tl.float32)
    mean = tl.sum(gf, axis=0) / N
    diff = gf - mean
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    gamma = tl.load(G_ptr + cols).to(tl.float32)
    beta = tl.load(B_ptr + cols).to(tl.float32)
    lnh = (diff * inv * gamma + beta).to(tl.float16)

    # bias add in fp16
    b4 = tl.load(B4_ptr + cols)
    y = lnh + b4

    # RMSNorm in fp32 as in reference, cast to fp16, then fp16 multiply by weight
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = yf * (1.0 / tl.sqrt(ms + 1e-6))
    rh = r.to(tl.float16)
    w = tl.load(W_ptr + cols)
    out = rh * w

    tl.store(OUT_ptr + row * N + cols, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        y = torch.matmul(x, self.W0)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        _fused_post_kernel[(Mrows,)](
            y, self.b1, self.ln3_g, self.ln3_b, self.b4, self.rms5_w, out,
            N=N, BLOCK=512,
            num_warps=4,
        )
        return out
