import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400002
S, D, N, E, DT = 512, 1024, 2048, 4, torch.float16


@triton.jit
def _gate_gelu_kernel(
    big_ptr,      # (S, E*N) fp16 : concatenated expert outputs
    gate_ptr,     # (S, E)   fp16 : softmax gate
    out_ptr,      # (S, N)   fp16
    N,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = big_ptr + pid_s * (E * N)
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e).to(tl.float32)
        v = tl.load(base + e * N + offs, mask=mask, other=0.0).to(tl.float32)
        acc += g * v

    # exact GELU (erf variant, matching F.gelu default)
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(out_ptr + pid_s * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback: reference implementation
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(self.We.shape[0])], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        e, d, n = self.We.shape

        # Lazily cache the flattened expert weight (D, E*N) so all expert
        # matmuls fuse into a single large GEMM (params are frozen).
        We_flat = getattr(self, "_We_flat", None)
        if We_flat is None or We_flat.device != x.device:
            We_flat = self.We.permute(1, 0, 2).reshape(d, e * n).contiguous()
            self._We_flat = We_flat

        x = x.contiguous()
        s = x.shape[0]

        # gate: (S, E)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # single big GEMM: (S, D) @ (D, E*N) -> (S, E*N)
        big = torch.matmul(x, We_flat)

        out = torch.empty((s, n), device=x.device, dtype=torch.float16)

        BLOCK = 1024
        grid = (s, triton.cdiv(n, BLOCK))
        _gate_gelu_kernel[grid](
            big, gate, out,
            n,
            E=e,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
