#!/usr/bin/env python3
"""Formal equivalence stats for the compounding null (panel #6: Astra "narrow the claim", Fable "say equivalence
formally"). Reads compounding_*/compounding.json lineage_minus_reset values (from S3-synced docs/data or a dir),
and reports, per model + pooled:
  * mean, 95% CI
  * TOST equivalence test at margin +/- delta (default 0.05): equivalent if the 90% CI lies within [-d,+d]
  * a JZS-style Bayes factor BF01 (H0: mu=0 vs H1: mu~Cauchy) via BIC approximation as a robust fallback
Prints a LaTeX-ready line. Pure stdlib (no scipy dependency on the cluster)."""
import json, glob, os, math, argparse, statistics as st


def tost(vals, delta, alpha=0.05):
    n = len(vals)
    if n < 2:
        return None
    m = st.mean(vals); sd = st.stdev(vals); se = sd / math.sqrt(n)
    # 90% CI (1-2alpha) for TOST at alpha each side
    z = 1.645  # ~ t_{0.95} for moderate n; conservative-ish. (normal approx; note df in paper)
    lo, hi = m - z * se, m + z * se
    equiv = (lo > -delta) and (hi < delta)
    # two one-sided test statistics
    t_lower = (m - (-delta)) / se   # H0: mu <= -delta
    t_upper = ((delta) - m) / se    # H0: mu >= +delta
    return {"n": n, "mean": m, "se": se, "ci90": (lo, hi), "equivalent": equiv,
            "t_lower": t_lower, "t_upper": t_upper, "delta": delta}


def bf01_bic(vals):
    """BIC approximation to BF01 (evidence for H0: mu=0 vs H1: mu!=0) for a one-sample mean.
    BF01 = sqrt(n) if the effect is null-ish; via BIC: BF01 = exp((BIC_H1 - BIC_H0)/2) inverse.
    Uses the one-sample t stat. >3 = moderate evidence for null, >10 = strong."""
    n = len(vals)
    if n < 2:
        return None
    m = st.mean(vals); sd = st.stdev(vals); se = sd / math.sqrt(n)
    if se == 0:
        return float("inf")
    t = m / se
    # BIC approximation (Wagenmakers 2007, eq. 11): BF01 = sqrt(n) * (1 + t^2/(n-1))^(-n/2).
    # t->0 gives BF01 -> sqrt(n) (evidence for the null grows with n); large |t| drives BF01 -> 0.
    bf01 = math.sqrt(n) * (1 + t * t / (n - 1)) ** (-n / 2.0)
    return bf01


def load_dir(d):
    groups = {}
    for f in sorted(glob.glob(os.path.join(d, "compounding_*", "compounding.json")) +
                    glob.glob(os.path.join(d, "*.json"))):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        model = (j.get("model") or os.path.basename(os.path.dirname(f))).split("/")[-1]
        key = "".join(c for c in model if not c.isdigit() or True)  # keep full name; group below by size tag
        vals = [h["lineage_minus_reset"] for h in j.get("history", []) if h.get("lineage_minus_reset") is not None]
        groups.setdefault(model, []).extend(vals)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data"))
    ap.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args()
    groups = load_dir(a.dir)
    if not groups:
        print("no compounding data found under", a.dir); return
    pooled = []
    print("model                         n   mean      90%%CI            TOST(d=%.2f)  BF01" % a.delta)
    for g in sorted(groups):
        v = groups[g]; pooled += v
        r = tost(v, a.delta); bf = bf01_bic(v)
        if r:
            print("%-28s %3d  %+.3f  [%+.3f,%+.3f]  %-9s  %.1f" %
                  (g[:28], r["n"], r["mean"], r["ci90"][0], r["ci90"][1],
                   "EQUIV" if r["equivalent"] else "not-equiv", bf if bf != float("inf") else 999))
    r = tost(pooled, a.delta); bf = bf01_bic(pooled)
    print("-" * 78)
    print("%-28s %3d  %+.3f  [%+.3f,%+.3f]  %-9s  %.1f" %
          ("POOLED", r["n"], r["mean"], r["ci90"][0], r["ci90"][1],
           "EQUIV" if r["equivalent"] else "not-equiv", bf if bf != float("inf") else 999))
    print("\nInterpretation: TOST EQUIV at delta=%.2f => compounding advantage is statistically bounded within +/-%.2f."
          % (a.delta, a.delta))
    print("BF01>3 => moderate evidence FOR the null (no effect) over a delta-sized effect.")


if __name__ == "__main__":
    main()
