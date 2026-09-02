"""Public / held-out split definition for KernelAscent.

PUBLIC dev split  : released in the repo (task.py + meta only) for self-benchmarking
                    and research papers. Anyone can regenerate it deterministically.
HELD-OUT test set : a PRIVATE seed range, never committed. Maintainers grade
                    leaderboard submissions on it, so public scores cannot be gamed
                    by overfitting to the released tasks.

Splits are separated by seed range so they never overlap (fusion tasks use seed0;
the attention grid uses seed0 + 100000 internally).
"""

PUBLIC = dict(n_fusion=150, seed0=0)              # released
HELDOUT = dict(n_fusion=300, seed0=10_000_000)    # PRIVATE — do not commit materialized tasks

SPLITS = {"public": PUBLIC, "heldout": HELDOUT}
