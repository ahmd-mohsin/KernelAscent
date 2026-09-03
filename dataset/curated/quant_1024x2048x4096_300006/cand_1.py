import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300006
M, D, N, DT = 1024, 2048, 4096, torch.float16


@triton.jit
def _int8_dequant_gemm_bias_gelu(
    A, B, SCALE, BIAS, C,
    M, N, K,
    sam, sak, sbk, sbn, scm, scn,
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
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    a_ptrs = A + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = B + rk[:, None] * sbk + rn[None, :] * sbn

    # per-column scale, cast to fp16 to match reference (scale.to(x.dtype))
    s = tl.load(SCALE + rn).to(tl.float16)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs).to(tl.float16) * s[None, :]  # fp16 dequant like reference
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk

    bi = tl.load(BIAS + rn)
    y = acc.to(tl.float16) + bi[None, :]          # fp16 bias add like reference
    yf = y.to(tl.float32)                          # gelu computed in fp32 (opmath)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    c_ptrs = C + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, g.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if (not x.is_cuda) or x.dtype != torch.float16 or x.dim() != 2:
            w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
            return F.gelu(x @ w + self.bias)

        x = x.contiguous()
        Mx, K = x.shape
        Kw, Nw = self.wq.shape

        BM, BN, BK = 128, 128, 64
        if (Mx % BM) or (Nw % BN) or (K % BK) or K != Kw:
            w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
            return F.gelu(x @ w + self.bias)

        out = torch.empty((Mx, Nw), device=x.device, dtype=torch.float16)
        wq = self.wq
        grid = (triton.cdiv(Mx, BM) * triton.cdiv(Nw, BN),)
        _int8_dequant_gemm_bias_gelu[grid](
            x, wq, self.scale, self.bias, out,
            Mx, Nw, K,
            x.stride(0), x.stride(1),
            wq.stride(0), wq.stride(1),
            out.stride(0), out.stride(1),
            BM=BM, BN=BN, BK=BK, GROUP=8,
            num_warps=8, num_stages=3,
        )
        return out
