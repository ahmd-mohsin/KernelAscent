#!/usr/bin/env python3
"""The regularized-RSI trust region must measure what it claims before a lambda sweep means anything.

A wrong KL term does not crash. It produces a plausible number for every lambda and a plausible
curve across them, and the sweep reads as a result. This project has already published one figure
that was an artifact of an estimator rather than of the models, so the estimator gets a test first.

Runs only where torch is importable (the container); skips cleanly on a laptop.
"""
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("SKIP -- torch not available here (run inside the container)")
    raise SystemExit(0)


def _token_logp(logits, tgt):
    """Must stay identical to lab_weight_rsi._token_logp."""
    return logits.gather(2, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def k3(lp, lpr):
    logr = (lpr - lp).clamp(-20, 20)
    return (logr.exp() - 1 - logr)


def main():
    torch.manual_seed(0)
    B, T, V = 2, 7, 500
    logits = torch.randn(B, T, V) * 2
    tgt = torch.randint(0, V, (B, T))

    # The point of _token_logp is to avoid materializing a second [B, T, vocab] tensor. It is
    # only worth having if it is exactly equal to the tensor it avoids.
    ref = F.log_softmax(logits, -1).gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(ref, _token_logp(logits, tgt), atol=1e-5)
    print("  ok  _token_logp == log_softmax + gather")

    # A trust region that penalizes a policy for equalling its own reference would shrink every
    # update regardless of divergence, which looks exactly like a working regularizer.
    lp = _token_logp(logits, tgt)
    assert k3(lp, lp).abs().max() < 1e-6
    print("  ok  identical policies give exactly zero")

    prev = 0.0
    for scale in (0.0, 0.25, 0.5, 1.0, 2.0):
        other = _token_logp(logits + torch.randn_like(logits) * scale, tgt)
        v = k3(lp, other).mean().item()
        assert v >= -1e-6, "k3 must be non-negative, got %r" % v
        assert v >= prev - 1e-6, "k3 must grow as policies separate"
        prev = v
    print("  ok  non-negative, and increases with divergence")

    # Against the exact KL on a vocabulary small enough to sum, with tokens drawn from pi_theta,
    # which is the regime it is used in.
    V2 = 12
    pa = torch.softmax(torch.randn(V2) * 1.5, -1)
    pb = torch.softmax(torch.randn(V2) * 1.5, -1)
    exact = (pa * (pa.log() - pb.log())).sum().item()
    idx = torch.multinomial(pa, 200000, replacement=True)
    est = k3(pa.log()[idx], pb.log()[idx]).mean().item()
    assert abs(est - exact) / exact < 0.05, "k3 %.4f vs exact %.4f" % (est, exact)
    print("  ok  estimates the true KL within 5%% (exact %.4f, k3 %.4f)" % (exact, est))

    print("\nPASS -- the trust-region term measures what it claims to")
    return 0


if __name__ == "__main__":
    sys.exit(main())
