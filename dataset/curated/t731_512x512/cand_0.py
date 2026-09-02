import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 731
M, D, DT = 512, 512, torch.float16


@triton.jit
def _softmax_bias_gelu_kernel(
    X_ptr, B_ptr, Y_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for fp16 inputs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # relu (no-op on softmax output, kept for exactness), cast to fp16 like PyTorch
    p = tl.maximum(p, 0.0)
    p16 = p.to(tl.float16)

    # bias add in fp16 (matches PyTorch fp16 add)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    h16 = p16 + b

    # exact (erf-based) GELU computed in fp32, cast back to fp16
    hf = h16.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))

    tl.store(Y_ptr + row * N + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            h = x @ self.W0
            h = torch.softmax(h, dim=-1)
            h = torch.relu(h)
            h = h + self.b3
            h = F.gelu(h)
            return h @ self.W5

        # GEMM 1 via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)

        _softmax_bias_gelu_kernel[(rows,)](
            h, self.b3, out, N,
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2 via cuBLAS
        return torch.matmul(out, self.W5)
