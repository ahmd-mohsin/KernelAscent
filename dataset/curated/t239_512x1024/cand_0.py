import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 239
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_gelu_softmax_rms2_bias(
    Y_ptr, W3_ptr, W4_ptr, B5_ptr, Out_ptr,
    stride_y, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    y = tl.load(Y_ptr + row * stride_y + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact / erf-based), computed in fp32 then rounded to fp16 (matches PyTorch half kernel)
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax in fp32, output rounded to fp16
    g_m = tl.where(mask, g, float("-inf"))
    mx = tl.max(g_m, axis=0)
    e = tl.exp(g_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # RMSNorm 1
    ms1 = tl.sum(sm * sm, axis=0) / N
    r1 = (sm * tl.math.rsqrt(ms1 + 1e-6)).to(tl.float16).to(tl.float32)
    w3 = tl.load(W3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = (r1 * w3).to(tl.float16).to(tl.float32)

    # RMSNorm 2
    ms2 = tl.sum(x1 * x1, axis=0) / N
    r2 = (x1 * tl.math.rsqrt(ms2 + 1e-6)).to(tl.float16).to(tl.float32)
    w4 = tl.load(W4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = (r2 * w4).to(tl.float16).to(tl.float32)

    # Bias add
    b = tl.load(B5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (x2 + b).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not (x.is_cuda and x.dtype == torch.float16):
            return self._forward_ref(x)

        # GEMM via cuBLAS (fp16 tensor cores, fp32 accumulate) - same as reference
        y = x @ self.W0
        y = y.contiguous()

        rows, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_softmax_rms2_bias[(rows,)](
            y, self.rms3_w, self.rms4_w, self.b5, out,
            y.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return out

    def _forward_ref(self, x):
        x = x @ self.W0
        x = F.gelu(x)
        x = torch.softmax(x, dim=-1)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
        x = x + self.b5
        return x
