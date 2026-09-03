import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300008
M, D, N, DT = 4096, 1024, 1024, torch.float16


@triton.jit
def _int8_gemm_gelu_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    mask_m = rm < M
    mask_n = rn < N

    # per-column scale, cast to fp16 to match reference dequant math
    scale = tl.load(scale_ptr + rn, mask=mask_n, other=0.0).to(tl.float16)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=mask_m[:, None] & (rk[None, :] < k_rem), other=0.0)
        w8 = tl.load(w_ptrs, mask=(rk[:, None] < k_rem) & mask_n[None, :], other=0)
        b = w8.to(tl.float16) * scale[None, :]
        acc += tl.dot(a, b)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # cast matmul result to fp16 (matches fp16 GEMM output), add fp16 bias
    y = acc.to(tl.float16)
    bias = tl.load(bias_ptr + rn, mask=mask_n, other=0.0).to(tl.float16)
    y = y + bias[None, :]

    # gelu computed in fp32 (matches PyTorch opmath for half)
    yf = y.to(tl.float32)
    out = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, out.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
            return F.gelu(x @ w + self.bias)

        x = x.contiguous()
        Mx, K = x.shape
        Kw, Nw = self.wq.shape
        out = torch.empty((Mx, Nw), device=x.device, dtype=torch.float16)

        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
        grid = (triton.cdiv(Mx, BLOCK_M) * triton.cdiv(Nw, BLOCK_N),)
        _int8_gemm_gelu_kernel[grid](
            x, self.wq, self.scale, self.bias, out,
            Mx, Nw, K,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=8, num_stages=4,
        )
        return out
