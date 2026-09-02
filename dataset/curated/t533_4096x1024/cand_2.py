import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 533
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_relu_rms_scale(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    s1, s2, s3,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu (in input dtype, exact)
    x = tl.maximum(x, 0.0)

    # float32 for rms
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # normalized, cast to bf16 (matches .to(x.dtype))
    y = (xf * inv).to(tl.bfloat16)

    # multiply by weight in bf16 semantics (fp32 compute, round to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # sequential scalar multiplies, each rounded to bf16 like PyTorch
    y = (y.to(tl.float32) * s1).to(tl.bfloat16)
    y = (y.to(tl.float32) * s2).to(tl.bfloat16)
    y = (y.to(tl.float32) * s3).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


def _bf16_scalar(v):
    # PyTorch converts python scalar to bf16 for bf16 tensor ops
    return float(torch.tensor(v, dtype=torch.bfloat16).float().item())


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        s1 = _bf16_scalar(1.1764)
        s2 = _bf16_scalar(1.103)
        s3 = _bf16_scalar(1.281)
        _fused_relu_rms_scale[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            n, 1e-6,
            s1, s2, s3,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
