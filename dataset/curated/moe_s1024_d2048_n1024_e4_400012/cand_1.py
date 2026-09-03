import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400012
S, D, N, E, DT = 1024, 2048, 1024, 4, torch.float16


@triton.jit
def _gate_mix_gelu_kernel(
    logits_ptr,   # (S, E) fp16 gate logits
    outs_ptr,     # (S, E, N) fp16 expert outputs (contiguous)
    y_ptr,        # (S, N) fp16 output
    N,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # softmax over the E gate logits of this row
    le = tl.arange(0, E)
    lg = tl.load(logits_ptr + s * E + le).to(tl.float32)
    m = tl.max(lg, 0)
    p = tl.exp(lg - m)
    p = p / tl.sum(p, 0)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = outs_ptr + s * E * N
    for e in tl.static_range(E):
        v = tl.load(base + e * N + offs, mask=mask, other=0.0).to(tl.float32)
        ge = tl.sum(tl.where(le == e, p, 0.0), 0)
        acc += ge * v

    # exact (erf-based) GELU, matching F.gelu default
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + s * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_wcat(self):
        # cache concatenated expert weights: (D, E*N), so all expert GEMMs
        # collapse into a single large GEMM
        wc = getattr(self, "_Wcat", None)
        if wc is None or wc.device != self.We.device or wc.dtype != self.We.dtype:
            wc = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()
            self._Wcat = wc
        return wc

    def forward(self, x):
        if not x.is_cuda:
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(self.We.shape[0])], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        x = x.contiguous()
        s = x.shape[0]
        e = self.We.shape[0]
        n = self.We.shape[2]

        # gate logits (small GEMM)
        logits = (x @ self.Wr).contiguous()          # (S, E)

        # all experts in one big GEMM
        wcat = self._get_wcat()                      # (D, E*N)
        outs = x @ wcat                              # (S, E*N) contiguous == (S, E, N)

        y = torch.empty((s, n), device=x.device, dtype=torch.float16)

        BLOCK = 1024
        grid = (s, triton.cdiv(n, BLOCK))
        _gate_mix_gelu_kernel[grid](
            logits, outs, y, n,
            E=e, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
