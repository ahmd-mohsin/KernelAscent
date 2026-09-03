import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300005
M, D, N, DT = 1024, 2048, 1024, torch.bfloat16


@triton.jit
def _int8_gemm_bias_gelu_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    m_mask = rm < M
    n_mask = rn < N

    # per-output-channel scale, cast to bf16 to match reference (scale.to(x.dtype))
    scale = tl.load(scale_ptr + rn, mask=n_mask, other=0.0).to(tl.bfloat16)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = (rk + k) < K
        a = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w8 = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        # dequantize in bf16 exactly as reference: wq.to(bf16) * scale_bf16
        b = w8.to(tl.bfloat16) * scale[None, :]
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # matmul output is bf16 in the reference; add bias in bf16
    y = acc.to(tl.bfloat16)
    bias = tl.load(bias_ptr + rn, mask=n_mask, other=0.0).to(tl.bfloat16)
    y = y + bias[None, :]

    # GELU (erf-based) computed in fp32, cast back to bf16 (matches PyTorch bf16 gelu)
    yf = y.to(tl.float32)
    out = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


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
        Mm, K = x.shape
        Kw, Nn = self.wq.shape
        out = torch.empty((Mm, Nn), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        grid = (triton.cdiv(Mm, BLOCK_M), triton.cdiv(Nn, BLOCK_N))
        _int8_gemm_bias_gelu_kernel[grid](
            x, self.wq, self.scale, self.bias, out,
            Mm, Nn, K,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )
        return out
