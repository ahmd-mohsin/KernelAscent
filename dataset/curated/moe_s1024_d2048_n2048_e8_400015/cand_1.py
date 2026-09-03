import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400015
S, D, N, E, DT = 1024, 2048, 2048, 8, torch.float16


@triton.jit
def _scale_expand_kernel(
    x_ptr, g_ptr, z_ptr,
    D: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr,
):
    # z[s, e*D + d] = gate[s, e] * x[s, d]
    s = tl.program_id(0)
    e = tl.program_id(1)
    pd = tl.program_id(2)
    offs = pd * BLOCK + tl.arange(0, BLOCK)

    gv = tl.load(g_ptr + s * E + e)
    xv = tl.load(x_ptr + s * D + offs)
    tl.store(z_ptr + s * (E * D) + e * D + offs, xv * gv)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference implementation)
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(E)], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        x = x.contiguous()
        s_len, d = x.shape
        e = self.Wr.shape[1]
        n = self.We.shape[2]

        # Router: gate = softmax(x @ Wr)  -> (S, E), small GEMM
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # Key identity:
        #   y[s] = sum_e gate[s,e] * (x[s] @ We[e])
        #        = concat_e(gate[s,e] * x[s]) @ reshape(We, (E*D, N))
        # Build Z = (S, E*D) with one fused Triton kernel, then a single big GEMM.
        z = torch.empty((s_len, e * d), device=x.device, dtype=x.dtype)

        BLOCK = 256
        grid = (s_len, e, triton.cdiv(d, BLOCK))
        _scale_expand_kernel[grid](x, gate, z, D=d, E=e, BLOCK=BLOCK)

        w_flat = self.We.reshape(e * d, n)  # zero-copy view (contiguous param)
        y = z @ w_flat  # one large tensor-core GEMM (fp32 accumulation)

        return F.gelu(y)
