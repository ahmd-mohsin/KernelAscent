import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 704
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_ln_gelu_rms_kernel(
    X, G, B, W, Y,
    N,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptr = X + row * N + cols

    # load matmul output (bf16) and apply relu in fp32 (relu is exact)
    x = tl.load(ptr).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # ---- LayerNorm (fp32 internals, like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + eps_ln)
    g = tl.load(G + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    y = d * rstd * g + b
    # round to bf16 like the reference output of layer_norm
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU (erf-based), fp32 opmath then bf16 round ----
    z = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (explicit fp32 like reference) ----
    ms = tl.sum(z * z, axis=0) / N
    r = z * tl.math.rsqrt(ms + eps_rms)
    r = r.to(tl.bfloat16).to(tl.float32)

    # multiply by weight (bf16 op with fp32 opmath -> bf16)
    w = tl.load(W + cols).to(tl.float32)
    o = (r * w).to(tl.bfloat16).to(tl.float32)

    # final scalar scale (bf16 op with fp32 opmath -> bf16)
    o = (o * scale).to(tl.bfloat16)

    tl.store(Y + row * N + cols, o)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core matmul
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        _fused_ln_gelu_rms_kernel[(rows,)](
            h, self.ln2_g, self.ln2_b, self.rms4_w, y,
            N,
            1e-5, 1e-6, 1.1698,
            BLOCK=1024,
            num_warps=8,
        )
        return y
