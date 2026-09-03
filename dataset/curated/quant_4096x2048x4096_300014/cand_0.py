import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300014
M, D, N, DT = 4096, 2048, 4096, torch.float16


@triton.jit
def _int8_gemm_bias_gelu(
    X, W, S, B, Y,
    Mdim, Ndim, Kdim,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(Mdim, BLOCK_M)
    num_pid_n = tl.cdiv(Ndim, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rm = tl.max_contiguous(tl.multiple_of(rm % Mdim, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn % Ndim, BLOCK_N), BLOCK_N)

    # per-column scale, applied in fp16 to match reference dequantization
    s = tl.load(S + rn).to(tl.float16)

    x_ptrs = X + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = W + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(Kdim, BLOCK_K)):
        x = tl.load(x_ptrs)
        wq = tl.load(w_ptrs)
        w = wq.to(tl.float16) * s[None, :]
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # matmul result rounds to fp16 (as in reference), bias added in fp16
    y16 = acc.to(tl.float16)
    b = tl.load(B + rn).to(tl.float16)
    y16 = y16 + b[None, :]

    # exact (erf-based) GELU computed in fp32, cast back to fp16
    f = y16.to(tl.float32)
    g = 0.5 * f * (1.0 + tl.math.erf(f * 0.7071067811865476))

    rm_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm_out[:, None] < Mdim) & (rn_out[None, :] < Ndim)
    y_ptrs = Y + rm_out[:, None] * stride_ym + rn_out[None, :] * stride_yn
    tl.store(y_ptrs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
            x = x @ w + self.bias
            return F.gelu(x)

        x = x.contiguous()
        Mdim, Kdim = x.shape
        Ndim = self.wq.shape[1]
        y = torch.empty((Mdim, Ndim), device=x.device, dtype=torch.float16)

        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
        grid = (triton.cdiv(Mdim, BLOCK_M) * triton.cdiv(Ndim, BLOCK_N),)
        _int8_gemm_bias_gelu[grid](
            x, self.wq, self.scale, self.bias, y,
            Mdim, Ndim, Kdim,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=8, num_stages=4,
        )
        return y
