import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 934
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _fused_relu_ln_softmax_gelu(
    X, G, B, Y,
    stride_x, stride_y,
    N,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ReLU (exact in fp16, value-preserving)
    x = tl.maximum(x, 0.0)

    # LayerNorm (fp32 accumulation, like PyTorch's fp16 LN)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # round to fp16 as reference LN outputs fp16
    y = y.to(tl.float16).to(tl.float32)

    # Softmax (fp32 accumulation)
    y_masked = tl.where(mask, y, float("-inf"))
    mx = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to fp16 as reference softmax outputs fp16
    p = p.to(tl.float16).to(tl.float32)

    # exact GELU
    out = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))

    tl.store(Y + row * stride_y + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = torch.softmax(x, dim=-1)
            return F.gelu(x)

        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        rows, N = h.shape[0], h.shape[1]
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_ln_softmax_gelu[(rows,)](
            h, self.ln2_g, self.ln2_b, out,
            h.stride(0), out.stride(0),
            N,
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
