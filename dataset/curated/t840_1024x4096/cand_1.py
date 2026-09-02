import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 840
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_softmax_gelu_rms_kernel(
    x_ptr, w_ptr, out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = x_ptr + row * N + offs

    # ---- softmax (fp32 math, like PyTorch's bf16 softmax) ----
    x = tl.load(ptr).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    denom = tl.sum(e, axis=0)
    s = e / denom
    # round to bf16 (softmax output dtype), then continue in fp32
    s = s.to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU (erf-based, fp32 opmath as PyTorch does for bf16) ----
    g = 0.5 * s * (1.0 + tl.math.erf(s * 0.7071067811865476))
    # round to bf16 (gelu output dtype)
    g = g.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm in fp32 on the bf16-rounded values ----
    ms = tl.sum(g * g, axis=0) / N
    r = g * tl.math.rsqrt(ms + 1e-6)
    r_bf16 = r.to(tl.bfloat16)

    # ---- multiply by weight in bf16 (bf16 * bf16 -> bf16) ----
    w = tl.load(w_ptr + offs)
    out = r_bf16 * w
    tl.store(out_ptr + row * N + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (tensor cores)
        h = x @ self.W0
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        # Fused softmax + gelu + rmsnorm + weight-scale in one kernel
        _fused_softmax_gelu_rms_kernel[(rows,)](
            h, self.rms3_w, out,
            N=N, BLOCK=N,
            num_warps=4,
        )

        # GEMM 2 + fused relu
        y = out @ self.W4
        return torch.relu_(y)
