import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300004
M, D, N, DT = 1024, 2048, 1024, torch.float16


@triton.jit
def _dequant_matmul_bias_gelu_kernel(
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = wq_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # per-output-column scale (fp16, matching reference dequant precision)
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b_i8 = tl.load(w_ptrs, mask=offs_k[:, None] < k_rem, other=0)
        # dequantize exactly as reference: int8 -> fp16, times fp16 scale, rounded to fp16
        b = (b_i8.to(tl.float16) * scale[None, :]).to(tl.float16)
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # matmul output rounded to fp16 (as torch fp16 matmul does), then fp16 bias add
    c = acc.to(tl.float16)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float16)
    c = c + bias[None, :]

    # exact (erf) GELU computed in fp32 (PyTorch opmath for half), rounded back to fp16
    cf = c.to(tl.float32)
    g = 0.5 * cf * (1.0 + tl.math.erf(cf * 0.7071067811865476))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, g.to(tl.float16), mask=mask)


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
        Mx, K = x.shape
        Kw, Nw = self.wq.shape
        out = torch.empty((Mx, Nw), device=x.device, dtype=torch.float16)

        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
        grid = (triton.cdiv(Mx, BLOCK_M) * triton.cdiv(Nw, BLOCK_N),)
        _dequant_matmul_bias_gelu_kernel[grid](
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
