import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 624
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_relu_gelu_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (bf16) -> fp32
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # relu (exact on bf16 values, safe in fp32)
    x = tl.maximum(x, 0.0)

    # exact (erf) gelu computed in fp32 (matches PyTorch opmath for bf16),
    # then round back to bf16 like F.gelu output
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # second relu is identity on non-negative gelu(relu(x)) values (exact)

    # rmsnorm in fp32
    gm = tl.where(mask, g, 0.0)
    ms = tl.sum(gm * gm, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)

    # round normalized value to bf16 (matches .to(x.dtype)), then multiply by weight
    y = (g * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        if not h.is_cuda:
            h = torch.relu(h)
            h = F.gelu(h)
            h = torch.relu(h)
            _xf = h.float()
            h = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms4_w
            return h

        orig_shape = h.shape
        h2 = h.reshape(-1, orig_shape[-1]).contiguous()
        rows, N = h2.shape
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        _fused_relu_gelu_rms_kernel[(rows,)](
            h2, self.rms4_w, out,
            h2.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out.reshape(orig_shape)
