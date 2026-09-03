import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300000
M, D, N, DT = 1024, 1024, 1024, torch.float16


@triton.jit
def _fused_i8_gemm_gelu(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    Mdim, Ndim, Kdim,
    sxm, sxk, swk, swn, som, son,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    # per-column scale, cast to fp16 to match reference (scale.to(x.dtype))
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < Ndim, other=0.0)
    scale_h = scale.to(tl.float16)

    x_ptrs = x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    w_ptrs = w_ptr + offs_k[:, None] * swk + offs_n[None, :] * swn

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, tl.cdiv(Kdim, BK)):
        k0 = k * BK
        a = tl.load(x_ptrs, mask=(offs_m[:, None] < Mdim) & ((k0 + offs_k)[None, :] < Kdim), other=0.0)
        wq = tl.load(w_ptrs, mask=((k0 + offs_k)[:, None] < Kdim) & (offs_n[None, :] < Ndim), other=0)
        # dequantize in fp16 exactly like: wq.to(fp16) * scale_fp16
        w = wq.to(tl.float16) * scale_h[None, :]
        acc += tl.dot(a, w)
        x_ptrs += BK * sxk
        w_ptrs += BK * swk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < Ndim, other=0.0)
    # cast accumulator to fp16, add fp16 bias (matches reference fp16 add)
    c = acc.to(tl.float16) + bias

    # exact GELU (erf form) computed in fp32, like PyTorch's half gelu (opmath float)
    cf = c.to(tl.float32)
    y = 0.5 * cf * (1.0 + tl.math.erf(cf * 0.7071067811865476))

    out_ptrs = out_ptr + offs_m[:, None] * som + offs_n[None, :] * son
    tl.store(out_ptrs, y.to(tl.float16), mask=(offs_m[:, None] < Mdim) & (offs_n[None, :] < Ndim))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mdim, Kdim = x.shape
        Ndim = self.wq.shape[1]
        out = torch.empty((Mdim, Ndim), device=x.device, dtype=x.dtype)

        BM, BN, BK = 64, 128, 64
        grid = (triton.cdiv(Mdim, BM), triton.cdiv(Ndim, BN))
        _fused_i8_gemm_gelu[grid](
            x, self.wq, self.scale, self.bias, out,
            Mdim, Ndim, Kdim,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BM=BM, BN=BN, BK=BK,
            num_warps=8, num_stages=4,
        )
        return out
