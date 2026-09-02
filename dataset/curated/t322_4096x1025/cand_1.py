import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 322
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_relu_ln_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    D_dim, stride_x, stride_y,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_dim

    # load row, relu, in fp32
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # layernorm (fp32 accumulation, like PyTorch)
    mean = tl.sum(x, axis=0) / D_dim
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D_dim
    inv_std = tl.rsqrt(var + eps_ln)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv_std * g + b

    # round to bf16 (layer_norm output dtype), then scale in bf16 semantics
    y_bf = y.to(tl.bfloat16)
    z_bf = (y_bf.to(tl.float32) * scale).to(tl.bfloat16)

    # rms norm in fp32
    zf = z_bf.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    ms = tl.sum(zf * zf, axis=0) / D_dim
    r = tl.rsqrt(ms + eps_rms)
    out_bf = (zf * r).to(tl.bfloat16)

    # multiply by rms weight (bf16 op computed in fp32, rounded to bf16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    res = (out_bf.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, res, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_ln_rms_kernel[(m,)](
            x2, self.ln1_g, self.ln1_b, self.rms3_w, y,
            d, x2.stride(0), y.stride(0),
            1e-5, 1e-6, 1.4215,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
