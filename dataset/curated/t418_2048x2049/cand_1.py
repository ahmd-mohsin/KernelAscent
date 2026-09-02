import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 418
M, D, DT = 2048, 2049, torch.float16


@triton.jit
def _fused_bias_ln_softmax2_kernel(
    x_ptr, b0_ptr, g_ptr, beta_ptr, out_ptr,
    N, stride_x, stride_o, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # bias add in fp16 (matches PyTorch fp16 arithmetic), then upcast
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0)
    x = x + b0
    xf = x.to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch)
    xf_masked = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf_masked, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + beta
    # round to fp16 (LayerNorm output dtype), then upcast for softmax
    y = y.to(tl.float16).to(tl.float32)

    # Softmax 1 (fp32 internal)
    y = tl.where(mask, y, float("-inf"))
    m1 = tl.max(y, axis=0)
    e1 = tl.exp(y - m1)
    s1 = tl.sum(tl.where(mask, e1, 0.0), axis=0)
    p1 = e1 / s1
    p1 = p1.to(tl.float16).to(tl.float32)

    # Softmax 2 (fp32 internal)
    p1 = tl.where(mask, p1, float("-inf"))
    m2 = tl.max(p1, axis=0)
    e2 = tl.exp(p1 - m2)
    s2 = tl.sum(tl.where(mask, e2, 0.0), axis=0)
    p2 = e2 / s2

    tl.store(out_ptr + row * stride_o + offs, p2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_bias_ln_softmax2_kernel[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, out,
            N, x2.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
