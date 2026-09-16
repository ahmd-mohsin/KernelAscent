#!/usr/bin/env python3
"""P3 — MUTATION-DSL TASK-BANK GENERATOR. Parametrically synthesizes hundreds of self-contained, gradeable
GPU-kernel tasks by crossing OP-FAMILY x SHAPE x DTYPE (+ per-family variant). Each emitted `source` matches the
grader contract (agent_bench.build_ref): defines DT, class Model(__init__(self, dt)) on cuda, get_inputs(), forward.
This (a) fixes the too-small-bank probe-AUC instability (29 -> ~400 tasks, split by FAMILY for leakage-resistant
CV) and (b) supplies the self-play author's structured mutation space (vs from-scratch prose). Writes a JSON bank
[{name, source, tier, family}]; point KA_KERNEL_BANK at it. Run the learnability gate next to keep solvable tasks.

Usage: python3 scripts/gen_task_dsl.py --out dataset/kernel_bank/kernel_tasks_dsl.json
"""
import os, json, argparse, itertools

HEAD = "import torch\nimport torch.nn as nn\nDT = torch.{dt}\nROWS, HIDDEN = {rows}, {hidden}\n\n"

# each family: name -> (init_body, forward_body). Params built with dtype `dt`; input [ROWS,HIDDEN] of DT.
FAMILIES = {
 "rmsnorm_gate": ("""self.eps={eps}
        self.w=nn.Parameter((1.0+0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """s=x.float(); v=s.pow(2).mean(-1,keepdim=True); s=s*torch.rsqrt(v+self.eps)
        s=s.to(x.dtype)*self.w; return s*torch.sigmoid(s)"""),
 "layernorm_affine": ("""self.eps={eps}
        self.w=nn.Parameter((1.0+0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))
        self.b=nn.Parameter((0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """s=x.float(); m=s.mean(-1,keepdim=True); v=s.var(-1,keepdim=True,unbiased=False)
        s=(s-m)*torch.rsqrt(v+self.eps); return (s.to(x.dtype)*self.w+self.b)"""),
 "softmax_row": ("""self.sc={eps}""",
   """s=(x.float()*self.sc); s=s-s.amax(-1,keepdim=True); e=s.exp(); return (e/e.sum(-1,keepdim=True)).to(x.dtype)"""),
 "gelu_mlp": ("""self.w1=nn.Parameter((0.02*torch.randn(HIDDEN,HIDDEN,generator=g)).to(dev,dt))""",
   """h=x@self.w1; return (0.5*h*(1.0+torch.tanh(0.7978845608*(h+{eps}*h*h*h))))"""),
 "matmul_bias": ("""self.w=nn.Parameter((0.02*torch.randn(HIDDEN,HIDDEN,generator=g)).to(dev,dt))
        self.b=nn.Parameter((0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """return torch.relu(x@self.w+self.b*{eps})"""),
 "fused_elemwise": ("""self.a=nn.Parameter((0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))
        self.b=nn.Parameter((0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """return torch.relu(x*self.a+self.b)*torch.sigmoid(x*{eps})"""),
 "sigmoid_gate": ("""self.w=nn.Parameter((1.0+0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """return (x*torch.sigmoid(x*{eps}))*self.w"""),
 "softplus_norm": ("""self.eps={eps}""",
   """s=torch.nn.functional.softplus(x.float()); return (s/(s.mean(-1,keepdim=True)+self.eps)).to(x.dtype)"""),
}

TMPL = HEAD + """class Model(nn.Module):
    def __init__(self, dt=DT, hidden=HIDDEN):
        super().__init__()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = torch.Generator().manual_seed(0)
        {init}
    def forward(self, x):
        {fwd}

def get_inputs():
    g = torch.Generator().manual_seed(1)
    return [(torch.randn(ROWS, HIDDEN, generator=g)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"), DT)]
"""

SHAPES = [(8192, 2048), (16384, 4096), (32768, 4096), (4096, 8192), (16384, 2048)]
DTYPES = ["float16", "bfloat16", "float32"]
EPS = {"rmsnorm_gate": [1e-5, 1e-6], "layernorm_affine": [1e-5, 1e-6], "softmax_row": [1.0, 0.5],
       "gelu_mlp": [0.044715, 0.05], "matmul_bias": [1.0, 0.5], "fused_elemwise": [1.0, 0.7],
       "sigmoid_gate": [1.0, 1.702], "softplus_norm": [1e-5, 1e-6]}


def gen(args):
    bank = []
    for fam, (initb, fwdb) in FAMILIES.items():
        for dt, (rows, hidden), eps in itertools.product(DTYPES, SHAPES, EPS[fam]):
            # matmul/gelu are O(n*h^2): cap hidden to avoid OOM/timeout
            if fam in ("gelu_mlp", "matmul_bias") and hidden > 4096: continue
            src = TMPL.format(dt=dt, rows=rows, hidden=hidden,
                              init=initb.format(eps=eps), fwd=fwdb.format(eps=eps))
            name = "dsl_%s_%s_%dx%d_%s" % (fam, dt.replace("float", "f"), rows, hidden, str(eps).replace(".", "p").replace("-", "m"))
            tier = "L2" if fam in ("gelu_mlp", "matmul_bias") else "L1"
            bank.append({"name": name, "source": src, "tier": tier, "family": fam})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(bank, open(args.out, "w"), indent=2)
    fams = {}
    for b in bank: fams[b["family"]] = fams.get(b["family"], 0) + 1
    print("wrote %d tasks -> %s" % (len(bank), args.out))
    print("by family:", fams)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/kernel_bank/kernel_tasks_dsl.json")
    gen(ap.parse_args())
