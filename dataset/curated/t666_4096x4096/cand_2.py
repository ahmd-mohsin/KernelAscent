import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 666
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_ln_softmax_kernel(
    X_ptr, G_ptr, B_ptr, B3_ptr, Out_ptr,
    stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x_ptrs = X_ptr + row * stride_row + offs

    # load row in fp32
    x = tl.load(x_ptrs).to(tl.float32)

    # ---- LayerNorm (stats in fp32, like PyTorch's half layer_norm) ----
    mean = tl.sum(x, axis=0) / BLOCK
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / BLOCK
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)
    y = diff * rstd * g + b
    # cast to fp16 (intermediate result between ops in reference), back to fp32
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax 1 (fp32 accumulate) ----
    m1 = tl.max(y, axis=0)
    e1 = tl.exp(y - m1)
    s1 = tl.sum(e1, axis=0)
    y = (e1 / s1).to(tl.float16)

    # ---- Add bias in fp16 (matches fp16 elementwise add) ----
    b3 = tl.load(B3_ptr + offs)
    z = (y + b3).to(tl.float32)

    # ---- Softmax 2 ----
    m2 = tl.max(z, axis=0)
    e2 = tl.exp(z - m2)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Out_ptr + row * stride_row + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_ln_softmax_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.b3, out,
            h.stride(0),
            1e-5,
            BLOCK=N,
            num_warps=8,
        )
        return out
