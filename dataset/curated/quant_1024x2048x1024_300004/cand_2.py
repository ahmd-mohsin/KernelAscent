import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300004
M, D, N, DT = 1024, 2048, 1024, torch.float16


@triton.jit
def _int8_gemm_bias_gelu_kernel(
    x_ptr, wq_ptr, scale_ptr, bias_ptr, out_ptr,
    Mdim, Ndim, Kdim,
    sxm, sxk, swk, swn, som, son,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    m_mask = rm < Mdim
    n_mask = rn < Ndim

    # scale is fp32 param; reference casts it to fp16 before multiplying
    scale = tl.load(scale_ptr + rn, mask=n_mask, other=0.0).to(tl.float16)

    x_ptrs = x_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    w_ptrs = wq_ptr + rk[:, None] * swk + rn[None, :] * swn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, Kdim, BLOCK_K):
        k_mask = (rk + k) < Kdim
        a = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        wq = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        # dequantize in fp16 exactly as reference: int8 -> fp16, * fp16 scale
        w = wq.to(tl.float16) * scale[None, :]
        acc = tl.dot(a, w, acc)
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk

    # matmul output rounded to fp16 (matches fp16 tensor-core GEMM output)
    y16 = acc.to(tl.float16)

    bias = tl.load(bias_ptr + rn, mask=n_mask, other=0.0)
    # bias add computed in fp32 (PyTorch opmath), rounded back to fp16
    t = (y16.to(tl.float32) + bias.to(tl.float32)).to(tl.float16)

    # exact GELU (erf form) computed in fp32, output fp16
    tf = t.to(tl.float32)
    g = 0.5 * tf * (1.0 + tl.math.erf(tf * 0.7071067811865476))

    out_ptrs = out_ptr + rm[:, None] * som + rn[None, :] * son
    tl.store(out_ptrs, g.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.wq = nn.Parameter(torch.randint(-127, 128, (D, N), generator=g).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.rand(N, generator=g) * 0.02 + 0.005, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(N, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mdim, Kdim = x.shape
        Ndim = self.wq.shape[1]

        out = torch.empty((Mdim, Ndim), device=x.device, dtype=torch.float16)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        grid = (triton.cdiv(Mdim, BLOCK_M), triton.cdiv(Ndim, BLOCK_N))

        _int8_gemm_bias_gelu_kernel[grid](
            x, self.wq, self.scale, self.bias, out,
            Mdim, Ndim, Kdim,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )
        return out
