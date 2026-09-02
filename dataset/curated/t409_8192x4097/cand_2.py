import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 409
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _softmax_scale_kernel(
    X, Y,
    stride_xm, stride_ym,
    N, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    denom = tl.sum(num, axis=0)
    y = num / denom
    # match PyTorch: softmax outputs bf16, then scalar mul computed in fp32
    y_bf = y.to(tl.bfloat16)
    out = (y_bf.to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _softmax_scale_kernel[(Mrows,)](
            h, out,
            h.stride(0), out.stride(0),
            N, 1.1318,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
