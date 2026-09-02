import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 353
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_ln_softmax_kernel(
    X_ptr, G_ptr, B_ptr, B5_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based) -- round to fp16 to match reference intermediate
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch's half layer_norm)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv_std * gamma + beta
    y = y.to(tl.float16).to(tl.float32)

    # Softmax (fp32 accumulation)
    row_max = tl.max(tl.where(mask, y, float('-inf')), axis=0)
    e = tl.exp(y - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom
    sm = sm.to(tl.float16).to(tl.float32)

    # scale + bias (fp16 rounding between ops to match reference)
    out = sm * SCALE
    out = out.to(tl.float16).to(tl.float32)
    b5 = tl.load(B5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = out + b5

    tl.store(Y_ptr + row * stride_y + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = torch.softmax(x, dim=-1)
            x = x * 1.2464
            x = x + self.b5
            return x

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])

        # matmul via cuBLAS tensor cores
        h = torch.matmul(x2, self.W0)
        h = h.contiguous()

        M_rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_gelu_ln_softmax_kernel[(M_rows,)](
            h, self.ln2_g, self.ln2_b, self.b5, out,
            N, h.stride(0), out.stride(0),
            EPS=1e-5,
            SCALE=1.2464,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return out.reshape(orig_shape[:-1] + (N,))


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
