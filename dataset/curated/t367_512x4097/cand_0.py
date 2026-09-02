import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 367
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_gelu_rms_softmax(
    X_ptr, W_ptr, Out_ptr,
    stride_x, stride_o,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # Load row (bf16) and upcast to fp32
    x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)

    # Exact GELU (erf variant), computed in fp32 then rounded to bf16
    # (matches PyTorch opmath behavior for bf16 inputs)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)
    gf = g_bf.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(gf * gf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    normed_bf = (gf * inv).to(tl.bfloat16)

    # Scale by weight in bf16 (matches reference bf16 * bf16 multiply)
    w = tl.load(W_ptr + offs)
    y_bf = normed_bf * w

    # Softmax in fp32, output bf16
    yf = y_bf.to(tl.float32)
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (bf16)
        h = x @ self.W0

        if not h.is_cuda:
            h = F.gelu(h)
            _xf = h.float()
            h = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms2_w
            return torch.softmax(h, dim=-1)

        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        _fused_gelu_rms_softmax[(rows,)](
            h, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N,
            BLOCK=2048,
            num_warps=8,
        )
        return out
