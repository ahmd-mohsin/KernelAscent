import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300002
M, D, N, DT = 1024, 1024, 4096, torch.float16


@triton.jit
def _int8_gemm_bias_gelu_kernel(
    x_ptr, w_ptr, s_ptr, b_ptr, out_ptr,
    M, N, K,
    sxm, sxk, swk, swn, som, son,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    n_mask = rn < N
    m_mask = rm < M

    # scale.to(fp16) as in reference, then upcast for the multiply (torch opmath)
    s = tl.load(s_ptr + rn, mask=n_mask, other=0.0).to(tl.float16).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_ptrs = x_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    w_ptrs = w_ptr + rk[:, None] * swk + rn[None, :] * swn

    for k in range(0, K, BLOCK_K):
        k_mask = rk + k < K
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        wq = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        # dequantize: fp16(int8) * fp16(scale) computed in fp32, rounded to fp16
        w = (wq.to(tl.float32) * s[None, :]).to(tl.float16)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk

    # matmul output rounds to fp16 (as x @ w would)
    y = acc.to(tl.float16).to(tl.float32)

    b = tl.load(b_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)
    y = y + b[None, :]
    # elementwise add rounds to fp16, gelu computed in fp32 (torch opmath behavior)
    y = y.to(tl.float16).to(tl.float32)

    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
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
        if not x.is_cuda or x.dtype != torch.float16:
            w = self.wq.to(x.dtype) * self.scale.to(x.dtype)
            x = x @ w + self.bias
            return F.gelu(x)

        x = x.contiguous()
        Mx, K = x.shape
        Kw, Nw = self.wq.shape
        out = torch.empty((Mx, Nw), device=x.device, dtype=torch.float16)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        grid = (triton.cdiv(Mx, BLOCK_M), triton.cdiv(Nw, BLOCK_N))
        _int8_gemm_bias_gelu_kernel[grid](
            x, self.wq, self.scale, self.bias, out,
            Mx, Nw, K,
            x.stride(0), x.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )
        return out
