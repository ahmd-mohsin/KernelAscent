import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 769
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _scale_bias_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # Emulate PyTorch half elementwise semantics: compute in fp32 (opmath),
    # round to fp16 after each op (matching separate kernel launches).
    t = (x.to(tl.float32) * S1).to(tl.float16)
    t = (t.to(tl.float32) + b.to(tl.float32)).to(tl.float16)
    t = (t.to(tl.float32) * S2).to(tl.float16)

    # Softmax in fp32 (matches PyTorch half softmax accumulation type)
    z = t.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM (same as reference matmul)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _scale_bias_softmax_kernel[(Mrows,)](
            h, self.b2, y,
            h.stride(0), y.stride(0),
            N,
            1.2814, 1.0299,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
