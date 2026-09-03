import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300008
M, D, N, DT = 4096, 1024, 1024, torch.float16


@triton.jit
def _int8_dq_gemm_gelu(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    sxm, sxk, swk, swn, som, son,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # per-output-column dequant scale (rand() produces fp32; ref casts to fp16)
    scale = tl.load(scale_ptr + rn, mask=rn < N, other=0.0).to(tl.float16)

    x_ptrs = x_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    w_ptrs = w_ptr + rk[:, None] * swk + rn[None, :] * swn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            x_ptrs,
            mask=(rm[:, None] < M) & (rk[None, :] + k * BLOCK_K < K),
            other=0.0,
        )
        wq = tl.load(
            w_ptrs,
            mask=(rk[:, None] + k * BLOCK_K < K) & (rn[None, :] < N),
            other=0,
        )
        # dequantize in fp16 exactly like: wq.to(fp16) * scale.to(fp16)
        w = wq.to(tl.float16) * scale[None, :]
        acc += tl.dot(a, w)
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk

    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    # ref: (x @ w + bias) rounds to fp16 first, then gelu uses fp32 opmath
    h16 = (acc + bias[None, :]).to(tl.float16)
    h = h16.to(tl.float32)
    y = 0.5 * h * (1.0 + tl.math.erf(h * 0.7071067811865476))

    out_ptrs = out_ptr + rm[:, None] * som + rn[None, :] * son
    tl.store(out_ptrs, y.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


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
        out = torch.empty((Mx, Nw), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
        grid = (triton.cdiv(Mx, BLOCK_M) * triton.cdiv(Nw, BLOCK_N),)
        _int8_dq_gemm_gelu[grid](
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
