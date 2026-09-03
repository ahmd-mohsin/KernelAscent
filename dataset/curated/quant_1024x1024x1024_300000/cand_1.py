import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 300000
M, D, N, DT = 1024, 1024, 1024, torch.float16


@triton.jit
def _int8_gemm_bias_gelu_kernel(
    x_ptr, wq_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    sxm, sxk, swk, swn, som, son,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # per-output-column scale (fp32 param cast to fp16, matching scale.to(x.dtype))
    scale = tl.load(scale_ptr + rn, mask=rn < N, other=0.0).to(tl.float16)

    x_ptrs = x_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    w_ptrs = wq_ptr + rk[:, None] * swk + rn[None, :] * swn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        xm = (rm[:, None] < M) & ((rk[None, :] + k) < K)
        wm = ((rk[:, None] + k) < K) & (rn[None, :] < N)
        x = tl.load(x_ptrs, mask=xm, other=0.0)
        wq = tl.load(w_ptrs, mask=wm, other=0)
        # dequantize in fp16: wq.to(fp16) * scale  (matches reference elementwise math)
        w = wq.to(tl.float16) * scale[None, :]
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk

    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float16)
    # cast accumulator to fp16 (matmul output dtype), add fp16 bias
    y = acc.to(tl.float16) + bias[None, :]
    # GELU computed in fp32 opmath (matches PyTorch half GELU kernel)
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    out = g.to(tl.float16)

    om = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptr + rm[:, None] * som + rn[None, :] * son, out, mask=om)


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

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).contiguous()
        Mm, K = x2.shape
        Nn = self.wq.shape[1]

        out = torch.empty((Mm, Nn), device=x.device, dtype=torch.float16)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
        grid = (triton.cdiv(Mm, BLOCK_M), triton.cdiv(Nn, BLOCK_N))
        _int8_gemm_bias_gelu_kernel[grid](
            x2, self.wq, self.scale, self.bias, out,
            Mm, Nn, K,
            x2.stride(0), x2.stride(1),
            self.wq.stride(0), self.wq.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )
        return out.reshape(*orig_shape[:-1], Nn)
