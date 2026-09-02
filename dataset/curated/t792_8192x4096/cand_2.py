import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 792
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_scale_relu_gelu_softmax(
    X, Y,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    ptr = X + row * n_cols + offs

    # Load fp16, upcast to fp32 for math
    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

    # x * 1.2833 (computed in fp32, rounded to fp16 like eager half mul)
    x = (x * 1.2833).to(tl.float16).to(tl.float32)
    # x * 1.3985
    x = (x * 1.3985).to(tl.float16).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # exact gelu (erf form), computed in fp32, rounded to fp16
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # softmax over the row (fp32 accumulation, like PyTorch half softmax)
    x = tl.where(mask, x, float("-inf"))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * n_cols + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = x * 1.2833
            x = x * 1.3985
            x = torch.relu(x)
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            return x

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]

        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_scale_relu_gelu_softmax[(n_rows,)](
            x2d, out, n_cols,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
