import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 524
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_scale_bias_relu_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # Replicate PyTorch bf16 opmath: compute in fp32, round back to bf16 each op
    x = (x * 1.2872).to(tl.bfloat16).to(tl.float32)
    x = (x * 1.4411).to(tl.bfloat16).to(tl.float32)

    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # ReLU (exact on bf16 values)
    x = tl.maximum(x, 0.0)

    # Softmax in fp32 (matches PyTorch bf16 softmax which upcasts to fp32)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out_ptr + row * stride_o + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # bf16 GEMM on tensor cores (same as reference)
        h = torch.matmul(x, self.W0)

        if not h.is_cuda:
            h = h * 1.2872
            h = h * 1.4411
            h = h + self.b3
            h = torch.relu(h)
            return torch.softmax(h, dim=-1)

        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_scale_bias_relu_softmax[(Mrows,)](
            h, self.b3, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
