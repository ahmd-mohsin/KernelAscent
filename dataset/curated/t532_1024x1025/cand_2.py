import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 532
M, D, DT = 1024, 1025, torch.bfloat16


@triton.jit
def _ln_fwd_kernel(
    X, G, B, Y,
    D,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    tl.store(Y + row * D + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _relu_scale_kernel(
    X,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    # relu in the tensor's dtype (identical values), then scale in fp32 opmath
    x = tl.maximum(x, 0.0)
    y = (x.to(tl.float32) * scale).to(X.dtype.element_ty)
    tl.store(X + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = x @ self.W1
            return torch.relu(x) * 1.2948

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]

        # Fused LayerNorm (fp32 stats, matches PyTorch bf16 layer_norm semantics)
        ln_out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _ln_fwd_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, ln_out,
            d, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        # cuBLAS bf16 GEMM (same as reference matmul path)
        out = torch.matmul(ln_out, self.W1)

        # Fused relu + scale (in-place on GEMM output)
        n = out.numel()
        EBLOCK = 1024
        _relu_scale_kernel[(triton.cdiv(n, EBLOCK),)](
            out, n, 1.2948,
            BLOCK=EBLOCK,
            num_warps=4,
        )

        return out.view(*orig_shape[:-1], out.shape[-1])
