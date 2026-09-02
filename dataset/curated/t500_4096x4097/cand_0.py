import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 500
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_add_ln_kernel(
    X_ptr, B1_ptr, G_ptr, B_ptr, Y_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X_ptr + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    row_max = tl.max(x, 0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, 0)
    sm = e / denom
    sm_bf = sm.to(tl.bfloat16)  # round to bf16 like PyTorch's softmax output

    # add bias: bf16 inputs, fp32 opmath, round back to bf16 (matches TensorIterator)
    b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0)
    v_bf = (sm_bf.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    v = v_bf.to(tl.float32)

    # layer norm: stats in fp32 over bf16 values (matches PyTorch layer_norm)
    v_masked = tl.where(mask, v, 0.0)
    mean = tl.sum(v_masked, 0) / D
    diff = tl.where(mask, v - mean, 0.0)
    var = tl.sum(diff * diff, 0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (v - mean) * rstd * g + b

    tl.store(Y_ptr + row * D + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_softmax_add_ln_kernel[(m,)](
            x2d, self.b1, self.ln2_g, self.ln2_b, y,
            D=d, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
