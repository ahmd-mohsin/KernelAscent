import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 582
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _ln_relu_gelu_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm statistics in fp32 (matches PyTorch's opmath for bf16)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # LayerNorm output is cast to bf16 in the reference before relu
    y_bf16 = y.to(tl.bfloat16)

    # ReLU on bf16 (exact)
    zero = tl.zeros_like(y_bf16)
    y_bf16 = tl.maximum(y_bf16, zero)

    # GELU (exact, erf-based) computed in fp32 like PyTorch's opmath
    f = y_bf16.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * f * (1.0 + tl.math.erf(f * INV_SQRT2))

    tl.store(Y_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)

        if not h.is_cuda:
            h = F.layer_norm(h, (h.shape[-1],), self.ln1_g, self.ln1_b)
            h = torch.relu(h)
            return F.gelu(h)

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        Mrows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _ln_relu_gelu_kernel[(Mrows,)](
            h2, self.ln1_g, self.ln1_b, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
