import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 539
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, out_ptr, n_cols, stride_row,
                  SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # x + b0 : compute in fp32, round to fp16 (matches fp16 elementwise add)
    v = (x.to(tl.float32) + b.to(tl.float32)).to(tl.float16)
    # x * 1.1541 : opmath fp32, round to fp16
    v = (v.to(tl.float32) * SCALE).to(tl.float16)
    # exact GELU: opmath fp32, round to fp16
    vf = v.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = vf * 0.5 * (1.0 + tl.math.erf(vf * INV_SQRT2))
    g = g.to(tl.float16)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    gf = g.to(tl.float32)
    gf = tl.where(mask, gf, float('-inf'))
    m = tl.max(gf, axis=0)
    e = tl.exp(gf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        if x.device.type != 'cuda':
            y = x + self.b0
            y = y * 1.1541
            y = F.gelu(y)
            return torch.softmax(y, dim=-1)
        rows, cols = x.shape[0], x.shape[-1]
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(cols)
        _fused_kernel[(rows,)](
            x, self.b0, out, cols, x.stride(0),
            SCALE=1.1541, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
