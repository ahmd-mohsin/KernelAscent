import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 31
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_scale_ln_kernel(
    X, G, B, Y,
    N,
    eps,
    s1, s2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # Emulate the two bf16 elementwise multiplies exactly:
    # PyTorch computes bf16 elementwise ops in fp32 (opmath) then rounds to bf16.
    x = (x * s1).to(tl.bfloat16).to(tl.float32)
    x = (x * s2).to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    # LayerNorm in fp32 (matches PyTorch's bf16 layer_norm which accumulates in fp32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.2676
            x = x * 1.0737
            return F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_scale_ln_kernel[(Mrows,)](
            x2d, self.ln2_g, self.ln2_b, y,
            N,
            1e-5,
            1.2676, 1.0737,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
