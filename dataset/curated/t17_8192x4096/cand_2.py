import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 17
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_bias_rms_softmax_ln(
    X_ptr, bias_ptr, rmsw_ptr, g_ptr, b_ptr, Y_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X_ptr + row * N + offs

    # x = x + b1  (bf16 elementwise: compute in fp32, round to bf16)
    x = tl.load(ptr).to(tl.float32)
    b = tl.load(bias_ptr + offs).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32, cast to bf16, multiply by rms2_w (bf16 result)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w = tl.load(rmsw_ptr + offs).to(tl.float32)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    x = (x * w).to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation, bf16 output)
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # layer_norm (fp32 accumulation, bf16 output)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(g_ptr + offs).to(tl.float32)
    bb = tl.load(b_ptr + offs).to(tl.float32)
    y = diff * inv * g + bb

    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        x = torch.matmul(x, self.W0)
        x = x.contiguous()
        rows, N = x.shape
        y = torch.empty_like(x)
        _fused_bias_rms_softmax_ln[(rows,)](
            x, self.b1, self.rms2_w, self.ln4_g, self.ln4_b, y,
            N=N, BLOCK=N, num_warps=8,
        )
        return y
