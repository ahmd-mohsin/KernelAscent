import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 568
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_gelu_rms_bias_kernel(
    X, W, B3, B4, Out,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load matmul output row (bf16 -> fp32 for math, matching PyTorch opmath)
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf variant), computed in fp32 then rounded to bf16 like F.gelu
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32 (matches _xf.pow(2).mean(-1) path)
    ms = tl.sum(g * g, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    n = (g * r).to(tl.bfloat16).to(tl.float32)

    # scale by rms weight (bf16 op with fp32 opmath, rounded to bf16)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (n * w).to(tl.bfloat16).to(tl.float32)

    # two sequential bias adds, each rounded to bf16 (matches x+b3 then +b4)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.bfloat16).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y + b4).to(tl.bfloat16)

    tl.store(Out + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # tensor-core matmul via cuBLAS (bf16 with fp32 accumulate)
        y = x @ self.W0

        if not y.is_cuda:
            # CPU fallback: reference path
            z = F.gelu(y)
            _zf = z.float()
            z = (_zf * torch.rsqrt(_zf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(z.dtype) * self.rms2_w
            z = z + self.b3
            z = z + self.b4
            return z

        orig_shape = y.shape
        d = orig_shape[-1]
        y2 = y.contiguous().view(-1, d)
        rows = y2.shape[0]
        out = torch.empty_like(y2)

        BLOCK = triton.next_power_of_2(d)
        _fused_gelu_rms_bias_kernel[(rows,)](
            y2, self.rms2_w, self.b3, self.b4, out,
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)
