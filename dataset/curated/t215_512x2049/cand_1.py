import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 215
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_relu_rms_relu(X, W, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu in bf16 (relu doesn't change values other than clamping)
    x = tl.maximum(x, 0.0)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + 1e-6)

    a = (xf * rs).to(tl.bfloat16)  # round to bf16, as in .to(x.dtype)
    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 on CUDA: computed in fp32, single rounding back to bf16
    y = (a.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS GEMM, bf16
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_rms_relu[(Mrows,)](
            x, self.rms2_w, y, N,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
