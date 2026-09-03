import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300010
M, D, N, DT = 4096, 1024, 4096, torch.float16


@triton.jit
def _int8_gemm_bias_gelu(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    Mm, Nn, Kk,
    sxm, sxk, swk, swn, som, son,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(Mm, BM)
    num_pid_n = tl.cdiv(Nn, BN)
    num_pid_in_group = GROUP * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(num_pid_m - first_pid_m, GROUP)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    m_mask = rm < Mm
    n_mask = rn < Nn

    x_ptrs = x_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    w_ptrs = w_ptr + rk[:, None] * swk + rn[None, :] * swn

    # scale is fp32; reference casts to x.dtype (fp16) before multiplying
    scale = tl.load(scale_ptr + rn, mask=n_mask, other=0.0).to(tl.float16)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(Kk, BK)):
        a = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        wq = tl.load(w_ptrs, mask=n_mask[None, :], other=0)
        w = wq.to(tl.float16) * scale[None, :]
        acc = tl.dot(a, w, acc)
        x_ptrs += BK * sxk
        w_ptrs += BK * swk

    bias = tl.load(bias_ptr + rn, mask=n_mask, other=0.0)
    # match reference: fp16 matmul result + fp16 bias, then gelu
    h = acc.to(tl.float16) + bias[None, :]
    hf = h.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    out = g.to(tl.float16)

    out_ptrs = out_ptr + rm[:, None] * som + rn[None, :] * son
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        BK = 64
        if (not x.is_cuda) or x.dtype != torch.float16 or (x.shape[1] % BK != 0):
            w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
            return F.gelu(x @ w + self.bias)

        x = x.contiguous()
        Mm, Kk = x.shape
        Nn = self.wq.shape[1]
        out = torch.empty((Mm, Nn), device=x.device, dtype=torch.float16)

        BM, BN, GROUP = 128, 128, 8
        grid = (triton.cdiv(Mm, BM) * triton.cdiv(Nn, BN),)
        _int8_gemm_bias_gelu[grid](
            x, self.wq, self.scale, self.bias, out,
            Mm, Nn, Kk,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BM=BM, BN=BN, BK=BK, GROUP=GROUP,
            num_warps=8, num_stages=4,
        )
        return out
