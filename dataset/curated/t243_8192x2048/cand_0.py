import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 243
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_ln_softmax_kernel(
    X, G, B, B4, Out,
    N: tl.constexpr,
    stride_x,
    eps,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch on bf16 inputs)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b

    # round to bf16 (matches intermediate storage in reference)
    y = y.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 math)
    y_masked = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom
    s = s.to(tl.bfloat16).to(tl.float32)

    # scale, round, add bias
    s = (s * scale).to(tl.bfloat16).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (s + b4).to(tl.bfloat16)

    tl.store(Out + row * stride_x + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS GEMM (bf16, tensor cores)
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.softmax(x, dim=-1)
            x = x * 1.1672
            x = x + self.b4
            return x

        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_ln_softmax_kernel[(m,)](
            x, self.ln1_g, self.ln1_b, self.b4, out,
            n, x.stride(0), 1e-5, 1.1672,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
