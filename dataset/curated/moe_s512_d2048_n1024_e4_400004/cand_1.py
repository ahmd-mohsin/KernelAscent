import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400004
S, D, N, E, DT = 512, 2048, 1024, 4, torch.float16


@triton.jit
def _gate_sum_gelu_kernel(
    outs_ptr,   # (E, S, N) fp16 contiguous
    gate_ptr,   # (S, E)   fp16 contiguous
    y_ptr,      # (S, N)   fp16
    S_size, N_size,
    E_num: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_size

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for e in range(E_num):
        g = tl.load(gate_ptr + pid_s * E_num + e).to(tl.float32)
        v = tl.load(outs_ptr + (e * S_size + pid_s) * N_size + offs,
                    mask=mask, other=0.0).to(tl.float32)
        acc += g * v

    # round to fp16 (matching reference y dtype) then exact-erf GELU
    yh = acc.to(tl.float16).to(tl.float32)
    out = 0.5 * yh * (1.0 + tl.math.erf(yh * 0.7071067811865476))

    tl.store(y_ptr + pid_s * N_size + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(self.We.shape[0])], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        # gating: (S, E) softmax (E is tiny; negligible cost)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # single batched GEMM over all experts: (S, D) @ (E, D, N) -> (E, S, N)
        outs = torch.matmul(x, self.We)

        E_num, S_size, N_size = outs.shape
        y = torch.empty((S_size, N_size), device=x.device, dtype=torch.float16)

        BLOCK = 1024
        grid = (S_size, triton.cdiv(N_size, BLOCK))
        _gate_sum_gelu_kernel[grid](
            outs, gate, y,
            S_size, N_size,
            E_num=E_num, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
