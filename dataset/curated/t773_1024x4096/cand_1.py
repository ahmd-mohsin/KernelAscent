import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 773
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_double_softmax_kernel(
    x_ptr, b_ptr, out_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- softmax #1 (fp32 accumulate, like PyTorch's bf16 softmax) ----
    x = tl.load(x_ptr + base + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m1 = tl.max(x, 0)
    e1 = tl.math.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    y = e1 / s1
    # round to bf16 exactly as PyTorch materializes the intermediate tensor
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- softmax #2 ----
    y = tl.where(mask, y, float('-inf'))
    m2 = tl.max(y, 0)
    e2 = tl.math.exp(y - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    z = e2 / s2
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- relu ----
    z = tl.maximum(z, 0.0)
    # relu output materialized as bf16 (values already bf16-exact, but keep semantics)
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- add bias (PyTorch bf16 add uses fp32 opmath, rounds to bf16) ----
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z + b).to(tl.bfloat16).to(tl.float32)

    # ---- scale (fp32 opmath, round to bf16) ----
    z = z * 1.3071

    tl.store(out_ptr + base + offs, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            y = torch.softmax(x, dim=-1)
            y = torch.softmax(y, dim=-1)
            y = torch.relu(y)
            y = y + self.b3
            return y * 1.3071

        x = x.contiguous()
        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.view(-1, d)
        n_rows = x2d.shape[0]

        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_double_softmax_kernel[(n_rows,)](
            x2d, self.b3, out,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
