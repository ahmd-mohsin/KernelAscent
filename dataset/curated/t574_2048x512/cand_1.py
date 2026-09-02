import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 574
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_gelu_ln_rms_softmax_rms(
    X, LNG, LNB, W3, W5, Y,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride + offs).to(tl.float32)

    # --- exact (erf-based) GELU, computed in fp32, rounded to bf16 like PyTorch ---
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # --- LayerNorm (fp32 accumulation, eps=1e-5), rounded to bf16 ---
    mean = tl.sum(g, axis=0) / N
    d = g - mean
    var = tl.sum(d * d, axis=0) / N
    ln = d * tl.math.rsqrt(var + 1e-5)
    gamma = tl.load(LNG + offs).to(tl.float32)
    beta = tl.load(LNB + offs).to(tl.float32)
    ln = ln * gamma + beta
    ln = ln.to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm #1 (fp32 mean of squares, eps=1e-6), cast to bf16, then bf16 mul by w3 ---
    ms = tl.sum(ln * ln, axis=0) / N
    r = ln * tl.math.rsqrt(ms + 1e-6)
    r = r.to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(W3 + offs).to(tl.float32)
    r = (r * w3).to(tl.bfloat16).to(tl.float32)

    # --- Softmax (fp32 internal like PyTorch), rounded to bf16 ---
    mx = tl.max(r, axis=0)
    e = tl.exp(r - mx)
    s = e / tl.sum(e, axis=0)
    s = s.to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm #2 ---
    ms2 = tl.sum(s * s, axis=0) / N
    r2 = s * tl.math.rsqrt(ms2 + 1e-6)
    r2 = r2.to(tl.bfloat16).to(tl.float32)
    w5 = tl.load(W5 + offs).to(tl.float32)
    out = (r2 * w5).to(tl.bfloat16)

    tl.store(Y + row * stride + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM on tensor cores via cuBLAS (bf16 in/out, matches reference matmul)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_ln_rms_softmax_rms[(m,)](
            h, self.ln2_g, self.ln2_b, self.rms3_w, self.rms5_w, y,
            n, h.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y
