import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 184
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_softmax_scale_relu_ln(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N,
    scale,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 internal, like torch), then round to bf16 like torch output
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = (e / denom).to(tl.bfloat16)

    # scale (opmath fp32, rounded back to bf16 like torch elementwise) + relu
    v = (sm.to(tl.float32) * scale).to(tl.bfloat16)
    v = tl.maximum(v, 0.0)
    vf = v.to(tl.float32)

    # layernorm in fp32
    mean = tl.sum(tl.where(mask, vf, 0.0), axis=0) / N
    diff = tl.where(mask, vf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = (vf - mean) * rstd * g + b
    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM

        if not h.is_cuda:
            h = torch.softmax(h, dim=-1)
            h = h * 1.2551
            h = torch.relu(h)
            return F.layer_norm(h, (h.shape[-1],), self.ln4_g, self.ln4_b)

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_softmax_scale_relu_ln[(rows,)](
            h2, self.ln4_g, self.ln4_b, out,
            N, 1.2551, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
