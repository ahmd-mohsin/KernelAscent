import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 44
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_bias_softmax_gelu_rms(
    Y_ptr, B_ptr, W_ptr, OUT_ptr,
    stride_y, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # load matmul result row and bias, add in fp32, round to fp16 (matches x + b1 in fp16)
    y = tl.load(Y_ptr + row * stride_y + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)
    t = (y + b).to(tl.float16).to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax with float accumulation), round to fp16
    m = tl.max(t, axis=0)
    e = tl.exp(t - m)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # exact GELU in fp32 (matches PyTorch half gelu opmath), round to fp16
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)

    # RMSNorm in fp32, round to fp16, then fp16 multiply by weight
    ms = tl.sum(gf * gf, axis=0) / BLOCK
    r = tl.math.rsqrt(ms + 1e-6)
    normed = (gf * r).to(tl.float16)
    w = tl.load(W_ptr + offs)
    out = normed * w

    tl.store(OUT_ptr + row * stride_o + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x @ self.W0
            x = x + self.b1
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        # GEMM via cuBLAS (tensor cores)
        y = x @ self.W0

        orig_shape = y.shape
        y2 = y.reshape(-1, orig_shape[-1])
        if not y2.is_contiguous():
            y2 = y2.contiguous()
        out = torch.empty_like(y2)

        n_rows = y2.shape[0]
        _fused_bias_softmax_gelu_rms[(n_rows,)](
            y2, self.b1, self.rms4_w, out,
            y2.stride(0), out.stride(0),
            BLOCK=512,
            num_warps=4,
        )
        return out.reshape(orig_shape)
