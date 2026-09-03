import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300001
M, D, N, DT = 1024, 1024, 1024, torch.bfloat16


@triton.jit
def _fused_int8_gemm_bias_gelu(
    x_ptr, wq_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    sxm, sxk, swk, swn, som, son,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    # scale: fp32 -> bf16 (matches scale.to(x.dtype)), then fp32 for exact-mult
    scale = tl.load(scale_ptr + rn, mask=rn < N, other=0.0)
    scale_bf = scale.to(tl.bfloat16).to(tl.float32)

    x_ptrs = x_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    w_ptrs = wq_ptr + rk[:, None] * swk + rn[None, :] * swn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(x_ptrs, mask=(rm[:, None] < M) & ((rk[None, :] + k) < K), other=0.0)
        wq = tl.load(w_ptrs, mask=((rk[:, None] + k) < K) & (rn[None, :] < N), other=0)
        # int8 -> bf16 is exact; multiply in fp32 then round to bf16
        # (identical to PyTorch's opmath behavior for bf16 elementwise mult)
        w = (wq.to(tl.float32) * scale_bf[None, :]).to(tl.bfloat16)
        acc = tl.dot(a, w, acc)
        x_ptrs += BK * sxk
        w_ptrs += BK * swk

    # matmul output rounds to bf16
    y = acc.to(tl.bfloat16).to(tl.float32)
    # bias add: compute in fp32, round to bf16 (PyTorch opmath for bf16)
    b = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    y = (y + b[None, :]).to(tl.bfloat16).to(tl.float32)
    # exact GELU in fp32, round to bf16
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    out_ptrs = out_ptr + rm[:, None] * som + rn[None, :] * son
    tl.store(out_ptrs, g.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() != 2:
            x = x.reshape(-1, orig_shape[-1])
        x = x.contiguous()
        Mx, K = x.shape
        Nw = self.wq.shape[1]

        out = torch.empty((Mx, Nw), device=x.device, dtype=x.dtype)

        BM, BN, BK = 64, 128, 64
        grid = (triton.cdiv(Mx, BM), triton.cdiv(Nw, BN))
        _fused_int8_gemm_bias_gelu[grid](
            x, self.wq, self.scale, self.bias, out,
            Mx, Nw, K,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BM=BM, BN=BN, BK=BK,
            num_warps=8, num_stages=4,
        )

        if len(orig_shape) != 2:
            out = out.reshape(*orig_shape[:-1], Nw)
        return out
