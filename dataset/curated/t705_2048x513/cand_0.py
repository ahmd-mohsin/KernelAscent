import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 705
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_bias_relu_softmax_ln(
    X_ptr, B1_ptr, B2_ptr, G_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0)
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0)

    # bias adds with bf16 rounding after each step (matches eager elementwise adds)
    x = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)

    # relu (exact on bf16)
    xf = tl.maximum(x.to(tl.float32), 0.0)

    # softmax in fp32 (matches torch's fp32 accumulation for bf16), round to bf16
    xf = tl.where(mask, xf, float('-inf'))
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p_bf16 = (e / s).to(tl.bfloat16)

    # layernorm in fp32 on bf16-rounded softmax output
    pf = p_bf16.to(tl.float32)
    mean = tl.sum(pf, axis=0) / N
    diff = tl.where(mask, pf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    tl.store(Out_ptr + row * stride_o + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (same as reference)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_relu_softmax_ln[(m,)](
            h, self.b1, self.b2, self.ln5_g, self.ln5_b, out,
            n, h.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
