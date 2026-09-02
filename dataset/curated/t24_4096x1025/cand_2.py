import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 24
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _scale_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    # replicate: x = x * 1.4032 (fp16 result), then float()
    xs = (x.to(tl.float32) * SCALE).to(tl.float16).to(tl.float32)

    ms = tl.sum(xs * xs, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)

    y16 = (xs * r).to(tl.float16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    out = y16 * w  # fp16 * fp16 as in reference

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.512 if False else torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0  # (M, 512), fp16

        M_, N_ = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _scale_rmsnorm_kernel[(M_,)](
            h, self.rms2_w, y,
            h.stride(0), y.stride(0),
            N=N_,
            EPS=1e-6,
            SCALE=1.4032,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # GEMM 2
        return y @ self.W3
