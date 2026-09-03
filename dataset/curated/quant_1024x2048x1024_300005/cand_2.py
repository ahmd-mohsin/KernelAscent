import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300005
M, D, N, DT = 1024, 2048, 1024, torch.bfloat16


@triton.jit
def _int8_deq_matmul_bias_gelu(
    A, Wq, Scale, Bias, Out,
    M, N, K,
    sam, sak, swk, swn, som, son,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(num_pid_m - first_pid_m, GROUP)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    n_mask = rn < N
    m_mask = rm < M

    # per-column scale, cast to bf16 to match reference: scale.to(x.dtype)
    scale = tl.load(Scale + rn, mask=n_mask, other=0.0).to(tl.bfloat16)

    a_ptrs = A + rm[:, None] * sam + rk[None, :] * sak
    w_ptrs = Wq + rk[:, None] * swk + rn[None, :] * swn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (rk[None, :] + k < K), other=0.0)
        w = tl.load(w_ptrs, mask=(rk[:, None] + k < K) & n_mask[None, :], other=0)
        # dequantize in bf16 exactly like: wq.to(bf16) * scale_bf16
        b = w.to(tl.bfloat16) * scale[None, :]
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        w_ptrs += BK * swk

    bias = tl.load(Bias + rn, mask=n_mask, other=0.0)  # bf16

    # match reference rounding: matmul result rounded to bf16, then + bias (bf16)
    y = acc.to(tl.bfloat16) + bias[None, :]

    # exact GELU (erf variant) computed in fp32
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    out = g.to(tl.bfloat16)

    out_ptrs = Out + rm[:, None] * som + rn[None, :] * son
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if (not x.is_cuda) or x.dtype != torch.bfloat16 or x.dim() != 2:
            w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
            return F.gelu(x @ w + self.bias)

        Mx, K = x.shape
        Nx = self.wq.shape[1]
        out = torch.empty((Mx, Nx), device=x.device, dtype=x.dtype)

        BM, BN, BK, GROUP = 128, 128, 64, 8
        grid = (triton.cdiv(Mx, BM) * triton.cdiv(Nx, BN),)
        _int8_deq_matmul_bias_gelu[grid](
            x, self.wq, self.scale, self.bias, out,
            Mx, Nx, K,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BM=BM, BN=BN, BK=BK, GROUP=GROUP,
            num_warps=8, num_stages=4,
        )
        return out
