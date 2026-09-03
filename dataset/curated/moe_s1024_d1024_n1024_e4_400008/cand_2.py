import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400008
S, D, N, E, DT = 1024, 1024, 1024, 4, torch.float16


@triton.jit
def _fused_moe_gelu_kernel(
    outs_ptr,      # (S, E*N) fp16 : concatenated expert outputs
    gate_ptr,      # (S, E)   fp32 : softmax gate
    y_ptr,         # (S, N)   fp16 : output
    N,
    stride_o, stride_g, stride_y,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * stride_g + e).to(tl.float32)
        o = tl.load(outs_ptr + pid_s * stride_o + e * N + offs_n,
                    mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + pid_s * stride_y + offs_n, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        We = self.We
        E_, D_, N_ = We.shape

        # Lazily cache the flattened expert weight (D, E*N) for one big GEMM
        We_flat = getattr(self, "_We_flat", None)
        if We_flat is None or We_flat.device != x.device:
            We_flat = We.permute(1, 0, 2).reshape(D_, E_ * N_).contiguous()
            self._We_flat = We_flat

        # Gate: small GEMM + softmax (compute softmax in fp32 for accuracy)
        gate = torch.softmax((x @ self.Wr).float(), dim=-1).contiguous()

        # All expert outputs in a single large GEMM: (S, D) @ (D, E*N)
        outs = x @ We_flat  # (S, E*N) fp16, contiguous

        S_ = x.shape[0]
        y = torch.empty((S_, N_), device=x.device, dtype=torch.float16)

        BLOCK_N = 1024
        grid = (S_, triton.cdiv(N_, BLOCK_N))
        _fused_moe_gelu_kernel[grid](
            outs, gate, y,
            N_,
            outs.stride(0), gate.stride(0), y.stride(0),
            E=E_, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
