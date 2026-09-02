import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 592
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_rms_ln_kernel(
    X_ptr, W1_ptr, G_ptr, B_ptr, Y_ptr,
    N, eps_rms, eps_ln,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (fp32 accumulation, matches reference)
    ms = tl.sum(x * x, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(ms + eps_rms)

    # cast to fp16 then multiply by fp16 weight in fp16 (matches reference dtype behavior)
    xh = (x * inv_rms).to(tl.float16)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0)
    t = (xh * w1).to(tl.float32)

    # LayerNorm (fp32 accumulation, matches PyTorch half layer_norm)
    mu = tl.sum(t, axis=0) / N
    d = tl.where(mask, t - mu, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y_ptr + row * N + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_rms_ln_kernel[(Mrows,)](
            h, self.rms1_w, self.ln2_g, self.ln2_b, y,
            N, 1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
