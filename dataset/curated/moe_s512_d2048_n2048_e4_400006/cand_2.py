import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400006
S, D, N, E, DT = 512, 2048, 2048, 4, torch.float16


@triton.jit
def _gate_gelu_kernel(
    outs_ptr,      # (S, E*N) fp16, expert outputs concatenated per row
    r_ptr,         # (S, E) fp16, router logits
    y_ptr,         # (S, N) fp16, output
    N,             # int
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    # softmax over the E router logits (fp32 for accuracy, matches torch softmax)
    e_offs = tl.arange(0, E)
    r = tl.load(r_ptr + pid_s * E + e_offs).to(tl.float32)
    rmax = tl.max(r, 0)
    ex = tl.exp(r - rmax)
    gate = ex / tl.sum(ex, 0)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    base = outs_ptr + pid_s * E * N
    for e in tl.static_range(E):
        v = tl.load(base + e * N + offs_n, mask=mask, other=0.0).to(tl.float32)
        g = tl.sum(tl.where(e_offs == e, gate, 0.0), 0)
        acc += g * v

    # exact GELU (erf-based)
    out = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + pid_s * N + offs_n, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Cache the experts fused into one big weight matrix: (D, E*N)
        Wcat = getattr(self, "_Wcat", None)
        if Wcat is None or Wcat.device != x.device:
            Wcat = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()
            self._Wcat = Wcat

        x = x.contiguous()
        s = x.shape[0]
        e = self.Wr.shape[1]
        n = self.We.shape[2]

        # One large GEMM for all experts + tiny GEMM for router logits
        outs = x @ Wcat                 # (S, E*N)
        r = x @ self.Wr                 # (S, E)
        r = r.contiguous()

        y = torch.empty((s, n), device=x.device, dtype=torch.float16)

        BLOCK_N = 1024
        grid = (s, triton.cdiv(n, BLOCK_N))
        _gate_gelu_kernel[grid](
            outs, r, y, n,
            E=e, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
