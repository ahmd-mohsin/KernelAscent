import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300007
M, D, N, DT = 1024, 2048, 4096, torch.bfloat16


@triton.jit
def _fused_i8_gemm_bias_gelu(
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

    rm_c = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rn_c = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    # per-output-column scale: replicate scale.to(bf16) then compute in fp32
    s = tl.load(scale_ptr + rn_c)  # fp32
    s_bf = s.to(tl.bfloat16)
    s_f = s_bf.to(tl.float32)

    x_ptrs = x_ptr + rm_c[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn_c[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=rk[None, :] < k_rem, other=0.0)          # bf16
        wq = tl.load(w_ptrs, mask=rk[:, None] < k_rem, other=0)           # int8
        # replicate: wq.to(bf16) * scale.to(bf16) -> bf16 (opmath fp32, rounded)
        w = (wq.to(tl.float32) * s_f[None, :]).to(tl.bfloat16)
        acc = tl.dot(a, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # matmul result stored as bf16 (matches x @ w output dtype)
    y = acc.to(tl.bfloat16)

    # bias add: bf16 + bf16 with fp32 opmath, rounded to bf16
    b = tl.load(bias_ptr + rn_c)  # bf16
    t = (y.to(tl.float32) + b.to(tl.float32)[None, :]).to(tl.bfloat16)

    # exact GELU (erf) with fp32 opmath, rounded to bf16
    tf = t.to(tl.float32)
    g = 0.5 * tf * (1.0 + tl.math.erf(tf * 0.7071067811865476))
    out = g.to(tl.bfloat16)

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 2:
            w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
            x = x @ w + self.bias
            return F.gelu(x)

        x = x.contiguous()
        Mx, K = x.shape
        Kw, Nw = self.wq.shape
        out = torch.empty((Mx, Nw), device=x.device, dtype=torch.bfloat16)

        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
        grid = (triton.cdiv(Mx, BLOCK_M) * triton.cdiv(Nw, BLOCK_N),)
        _fused_i8_gemm_bias_gelu[grid](
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
