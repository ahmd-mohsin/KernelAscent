import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 333
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _scale_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    scale, eps,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)

    x = tl.load(X_ptr + row * stride_x + offs)          # bf16
    # x = x * 1.4797 (computed in fp32, rounded to bf16 like PyTorch)
    xs_bf16 = (x.to(tl.float32) * scale).to(tl.bfloat16)
    xf = xs_bf16.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / D
    inv = tl.math.rsqrt(ms + eps)

    xn = (xf * inv).to(tl.bfloat16)                     # cast to bf16 as in reference
    w = tl.load(W_ptr + offs)                           # bf16
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul on tensor cores
        y = x @ self.W0

        if not y.is_cuda:
            y = y * 1.4797
            _yf = y.float()
            return (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms2_w

        y = y.contiguous()
        out = torch.empty_like(y)
        n_rows, d = y.shape
        _scale_rmsnorm_kernel[(n_rows,)](
            y, self.rms2_w, out,
            y.stride(0), out.stride(0),
            1.4797, 1e-6,
            D=d,
            num_warps=8,
        )
        return out


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
