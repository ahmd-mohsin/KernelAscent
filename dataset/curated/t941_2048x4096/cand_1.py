import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 941
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _ln_relu_kernel(
    X, G, B, Y,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    # ReLU commutes with rounding to bf16 (sign-preserving), so apply in fp32
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = torch.relu(x)
            x = x @ self.W2
            x = x * 1.007
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]

        h = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _ln_relu_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, h,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        out = torch.matmul(h, self.W2)
        out.mul_(1.007)
        return out.view(*orig_shape[:-1], self.W2.shape[1])
