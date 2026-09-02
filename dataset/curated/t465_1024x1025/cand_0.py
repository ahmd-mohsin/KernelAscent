import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 465
M, D, DT = 1024, 1025, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load fp16 row
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf'))
    # emulate torch: half tensor * python float -> compute in fp32, round to fp16
    xf = x.to(tl.float32) * SCALE
    xf = xf.to(tl.float16).to(tl.float32)

    # softmax with fp32 accumulation (matches torch half softmax semantics)
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            h = torch.relu(x @ self.W0)
            h = (h @ self.W2) * 1.0685
            return torch.softmax(h, dim=-1)

        # GEMM 1 (cuBLAS tensor cores) + in-place ReLU
        h = torch.mm(x, self.W0)
        h.relu_()

        # GEMM 2 (cuBLAS tensor cores)
        y = torch.mm(h, self.W2)

        # Fused scale + softmax in Triton (fp32 accumulation, matches torch)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _scale_softmax_kernel[(Mrows,)](
            y, out,
            N, y.stride(0), out.stride(0),
            SCALE=1.0685,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
