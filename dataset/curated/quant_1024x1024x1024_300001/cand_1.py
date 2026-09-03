import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300001
M, D, N, DT = 1024, 1024, 1024, torch.bfloat16


@triton.jit
def _fused_dequant_gemm_bias_gelu(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    mask_m = rm < M
    mask_n = rn < N

    # scale.to(bf16) (matches scale.to(x.dtype)), then use as fp32 for the mul
    scale = tl.load(scale_ptr + rn, mask=mask_n, other=0.0)
    scale_bf16 = scale.to(tl.bfloat16)
    scale_f32 = scale_bf16.to(tl.float32)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, K, BK):
        k_mask = (rk + k) < K
        a = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        wq = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0)
        # wq.to(bf16) is exact for int8; PyTorch bf16 mul computes in fp32,
        # then rounds result to bf16
        wf = wq.to(tl.float32) * scale_f32[None, :]
        wb = wf.to(tl.bfloat16)
        acc = tl.dot(a, wb, acc)
        x_ptrs += BK * stride_xk
        w_ptrs += BK * stride_wk

    # matmul output rounds to bf16 (like cuBLAS bf16 GEMM with fp32 accumulate)
    y = acc.to(tl.bfloat16).to(tl.float32)

    bias = tl.load(bias_ptr + rn, mask=mask_n, other=0.0).to(tl.float32)
    # bf16 add computed in fp32, rounded to bf16 (PyTorch opmath behavior)
    y = (y + bias[None, :]).to(tl.bfloat16).to(tl.float32)

    # exact erf-based GELU, computed in fp32, output bf16
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    out = g.to(tl.bfloat16)

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


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
            x = x @ w + self.bias
            return F.gelu(x)

        x = x.contiguous()
        Mx, K = x.shape
        Kw, Nw = self.wq.shape
        out = torch.empty((Mx, Nw), device=x.device, dtype=x.dtype)

        BM, BN, BK = 64, 128, 64
        grid = (triton.cdiv(Mx, BM), triton.cdiv(Nw, BN))
        _fused_dequant_gemm_bias_gelu[grid](
            x, self.wq, self.scale, self.bias, out,
            Mx, Nw, K,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BM=BM, BN=BN, BK=BK,
            num_warps=8, num_stages=4,
        )
        return out
