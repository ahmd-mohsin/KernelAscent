import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300003
M, D, N, DT = 1024, 1024, 4096, torch.bfloat16


@triton.jit
def _int8_gemm_bias_gelu_kernel(
    x_ptr, wq_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
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

    # scale is fp32 parameter; reference casts to bf16 before multiplying
    scale = tl.load(scale_ptr + rn).to(tl.bfloat16)

    x_ptrs = x_ptr + rm[:, None] * K + rk[None, :]
    w_ptrs = wq_ptr + rk[:, None] * N + rn[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(x_ptrs)                      # bf16
        wq = tl.load(w_ptrs)                     # int8
        w = wq.to(tl.bfloat16) * scale[None, :]  # dequant in bf16 (matches ref)
        acc = tl.dot(a, w, acc)                  # fp32 accumulate
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * N

    bias = tl.load(bias_ptr + rn)                # bf16
    # ref: (x @ w) -> bf16 result, then bf16 add with bias
    y = acc.to(tl.bfloat16) + bias[None, :]
    # ref F.gelu on bf16 upcasts to fp32 internally: 0.5*x*(1+erf(x/sqrt(2)))
    yf = y.to(tl.float32)
    g = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    out_ptrs = out_ptr + rm[:, None] * N + rn[None, :]
    tl.store(out_ptrs, g.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        K, Nn = self.wq.shape
        if (not x.is_cuda) or x.dtype != torch.bfloat16 or x.dim() != 2 \
                or x.shape[1] != K or (x.shape[0] % 64) or (Nn % 128) or (K % 64):
            w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
            return F.gelu(x @ w + self.bias)

        x = x.contiguous()
        Mm = x.shape[0]
        out = torch.empty((Mm, Nn), device=x.device, dtype=torch.bfloat16)

        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8
        grid = (triton.cdiv(Mm, BLOCK_M) * triton.cdiv(Nn, BLOCK_N),)
        _int8_gemm_bias_gelu_kernel[grid](
            x, self.wq, self.scale, self.bias, out,
            Mm, Nn, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
            num_warps=8, num_stages=4,
        )
        return out
