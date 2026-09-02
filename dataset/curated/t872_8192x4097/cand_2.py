import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 872
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_scale_rms_relu_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_x, stride_o,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    # replicate: _xf = (x * 1.25).float()
    xf = x.to(tl.float32) * SCALE

    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + EPS)

    # (_xf * rsqrt(...)).to(bf16)
    y_bf = (xf * rstd).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    # bf16 * bf16 -> PyTorch computes in fp32 (opmath) then rounds to bf16
    z = (y_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # relu
    zero = tl.zeros_like(z)
    out = tl.maximum(z, zero)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # LayerNorm (cuDNN/ATen fused kernel, fp32 accumulate internally - exact match)
        x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)

        # bf16 GEMM on tensor cores
        x = x @ self.W1

        # Fused: scale by 1.25 + RMSNorm + weight mul + ReLU in one Triton kernel
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_scale_rms_relu_kernel[(Mrows,)](
            x, self.rms3_w, out,
            x.stride(0), out.stride(0),
            N=N, EPS=1e-6, SCALE=1.25,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
