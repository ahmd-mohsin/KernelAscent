import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 734
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _ln_relu_kernel(
    X_ptr, Y_ptr, G_ptr, B_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    y = tl.maximum(y, 0.0)

    tl.store(Y_ptr + row * N + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = torch.matmul(x, self.W0)

        if h.is_cuda:
            h = h.contiguous()
            orig_shape = h.shape
            N = orig_shape[-1]
            h2 = h.view(-1, N)
            rows = h2.shape[0]
            out = torch.empty_like(h2)

            BLOCK = triton.next_power_of_2(N)
            num_warps = 8 if BLOCK >= 2048 else 4

            _ln_relu_kernel[(rows,)](
                h2, out, self.ln1_g, self.ln1_b,
                N, 1e-5,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
            h = out.view(orig_shape)
        else:
            h = F.layer_norm(h, (h.shape[-1],), self.ln1_g, self.ln1_b)
            h = torch.relu(h)

        # GEMM 2 (cuBLAS tensor cores)
        return torch.matmul(h, self.W3)
