import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300011
M, D, N, DT = 4096, 1024, 4096, torch.bfloat16


@triton.jit
def _dequant_gemm_bias_gelu_kernel(
    x_ptr, wq_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = wq_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    n_mask = offs_n < N
    # scale is stored fp32; reference casts to bf16 before multiplying
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.bfloat16)
    scale_f = scale.to(tl.float32)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=m_mask & (offs_k[None, :] < k_rem), other=0.0)
        wq = tl.load(w_ptrs, mask=(offs_k[:, None] < k_rem) & n_mask[None, :], other=0)
        # dequantize: int8 -> bf16 (exact) * bf16 scale, rounded to bf16 (matches reference)
        w = (wq.to(tl.float32) * scale_f).to(tl.bfloat16)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # matmul output rounded to bf16 (matches cuBLAS bf16 gemm output)
    c = acc.to(tl.bfloat16)
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    # bf16 add computed at fp32 opmath, rounded back to bf16
    xf = (c.to(tl.float32) + bias.to(tl.float32)[None, :]).to(tl.bfloat16).to(tl.float32)
    # exact (erf-based) GELU at fp32 opmath
    y = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    y = y.to(tl.bfloat16)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y, mask=m_mask & n_mask[None, :])


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
        Nw = self.wq.shape[1]
        out = torch.empty((Mx, Nw), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
        grid = (triton.cdiv(Mx, BLOCK_M) * triton.cdiv(Nw, BLOCK_N),)
        _dequant_gemm_bias_gelu_kernel[grid](
            x, self.wq, self.scale, self.bias, out,
            Mx, Nw, K,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
            num_warps=8, num_stages=4,
        )
        return out
