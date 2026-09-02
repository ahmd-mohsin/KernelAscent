import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 368
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, out_ptr,
    ln1_g_ptr, ln1_b_ptr, b2_ptr, ln3_g_ptr, ln3_b_ptr,
    N, stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, 0.0)

    # layernorm 1 (fp32 accumulation, like PyTorch on fp16 input)
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g1 = tl.load(ln1_g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bta1 = tl.load(ln1_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g1 + bta1
    # PyTorch would materialize this as fp16, then add b2 in fp16
    y = y.to(tl.float16)

    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)
    y = y + b2  # fp16 add

    # layernorm 2 on fp16 input, fp32 accumulation
    y32 = tl.where(mask, y.to(tl.float32), 0.0)
    mean2 = tl.sum(y32, axis=0) / N
    d2 = tl.where(mask, y32 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)

    g3 = tl.load(ln3_g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bta3 = tl.load(ln3_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g3 + bta3

    # relu
    z = tl.maximum(z, 0.0)

    tl.store(out_ptr + row * stride_row + cols, z.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = x + self.b2
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return torch.relu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(rows,)](
            x2, out,
            self.ln1_g, self.ln1_b, self.b2, self.ln3_g, self.ln3_b,
            N, x2.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
