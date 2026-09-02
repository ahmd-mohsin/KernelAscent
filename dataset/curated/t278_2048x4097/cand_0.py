import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 278
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_relu_scale_bias_rmsnorm(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (bf16), relu (relu(relu(x)) == relu(x))
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # x * 1.3017 computed in fp32, rounded to bf16 (matches PyTorch opmath behavior)
    x = (x * 1.3017).to(tl.bfloat16).to(tl.float32)

    # + bias, computed in fp32, rounded to bf16
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y = (x * r).to(tl.bfloat16).to(tl.float32)

    # * rms weight (fp32 opmath, round to bf16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 4096, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = x @ self.W0
        y = y.contiguous()
        rows = y.shape[0]
        n = y.shape[1]
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_relu_scale_bias_rmsnorm[(rows,)](
            y, self.b4, self.rms5_w, out,
            n, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
