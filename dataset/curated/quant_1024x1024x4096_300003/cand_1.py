import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300003
M, D, N, DT = 1024, 1024, 4096, torch.bfloat16


@triton.jit
def _int8_dequant_gemm_bias_gelu(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
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

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    # scale.to(x.dtype): fp32 -> bf16, then w = wq.to(bf16) * scale_bf16 (bf16 multiply)
    scale = tl.load(scale_ptr + rn).to(tl.bfloat16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(x_ptrs)
        wq = tl.load(w_ptrs)
        w = wq.to(tl.bfloat16) * scale[None, :]
        acc = tl.dot(a, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # matmul output rounds to bf16, then bf16 add (fp32 opmath) rounds to bf16
    bias = tl.load(bias_ptr + rn).to(tl.float32)
    s = acc.to(tl.bfloat16).to(tl.float32) + bias
    s = s.to(tl.bfloat16).to(tl.float32)

    # exact erf-based GELU in fp32 (matches F.gelu opmath on bf16)
    g = 0.5 * s * (1.0 + tl.math.erf(s * 0.7071067811865476))
    out = g.to(tl.bfloat16)

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        Mx, K = x.shape
        Kw, Nw = self.wq.shape

        if (not x.is_cuda) or (Mx % BLOCK_M) or (Nw % BLOCK_N) or (K % BLOCK_K) or K != Kw:
            w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
            y = x @ w + self.bias
            return F.gelu(y)

        x = x.contiguous()
        wq = self.wq
        out = torch.empty((Mx, Nw), device=x.device, dtype=x.dtype)

        grid = (triton.cdiv(Mx, BLOCK_M) * triton.cdiv(Nw, BLOCK_N),)
        _int8_dequant_gemm_bias_gelu[grid](
            x, wq, self.scale, self.bias, out,
            Mx, Nw, K,
            x.stride(0), x.stride(1),
            wq.stride(0), wq.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=8,
            num_warps=8, num_stages=4,
        )
        return out
