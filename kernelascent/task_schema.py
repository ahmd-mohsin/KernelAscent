"""KernelAscent task schema.

A Task is a self-contained, blindly-gradable problem: a reference implementation,
an input generator, a correctness tolerance, and metadata (tier, family, primitive
tags). The harness never needs to know how a task was made.
"""
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class Task:
    name: str
    tier: str                 # L0..L5
    family: str               # e.g. "norm-act", "matmul"
    tags: List[str]           # optimization primitives an expert solution needs
    ref: Callable             # ref(*inputs) -> Tensor  (defines correctness)
    make_inputs: Callable     # () -> tuple[Tensor]  (deterministic, fresh each call)
    atol: float = 1e-2
    rtol: float = 1e-2
    meta: dict = field(default_factory=dict)
