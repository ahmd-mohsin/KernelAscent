import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400014
S, D, N, E, DT = 1024, 2048, 2048, 4, torch.float16


@triton.jit
def _combine_gelu_kernel(
    logits_ptr,   # [S, E] fp16 gate logits
    outs_ptr,     # [S, E*N] fp16 expert outputs (row-major, expert-major within row)
    y_ptr,        # [S, N] fp16 output
    N, EN,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # softmax over the E logits of this row (fp32)
    l = tl.load(logits_ptr + pid_s * E + tl.arange(0, E)).to(tl.float32)
    m = tl.max(l, 0)
    p = tl.exp(l - m)
    denom = tl.sum(p, 0)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = outs_ptr + pid_s * EN
    for e in tl.static_range(E):
        le = tl.load(logits_ptr + pid_s * E + e).to(tl.float32)
        w = tl.exp(le - m) / denom
        o = tl.load(base + e * N + offs, mask=mask, other=0.0).to(tl.float32)
        acc += w * o

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    out = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + pid_s * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_weight(self):
        # [E, D, N] -> [D, E*N] so all experts run in ONE big GEMM
        w = getattr(self, "_W2", None)
        if w is None or w.device != self.We.device:
            w = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()
            self._W2 = w
        return w

    def forward(self, x):
        if not x.is_cuda:
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(self.We.shape[0])], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        e_num, d, n = self.We.shape
        s = x.shape[0]

        W2 = self._get_fused_weight()

        logits = torch.matmul(x, self.Wr)          # [S, E]  (tiny GEMM)
        outs = torch.matmul(x, W2)                 # [S, E*N] (single large GEMM)

        y = torch.empty((s, n), device=x.device, dtype=torch.float16)

        BLOCK = 1024
        grid = (s, triton.cdiv(n, BLOCK))
        _combine_gelu_kernel[grid](
            logits, outs, y,
            n, e_num * n,
            E=e_num, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
