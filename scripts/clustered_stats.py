#!/usr/bin/env python3
"""Pure-stdlib clustered inference for KernelAscent's longitudinal contrasts.

WHY THIS EXISTS (P0.1). Every headline n in the paper used to be a count of ROUND-COMPARISONS,
but rounds inside one trajectory share a base checkpoint, a seed, a held split and an accumulated
adapter, so they are nowhere near independent. Treating them as independent shrinks the standard
error by ~sqrt(m) (m = rounds per trajectory) and inflates BF01. The independent unit is the
TRAJECTORY (one run = one seed = one lineage), not the round.

This module supplies the three estimands we report side by side:

  round      naive round-level pooling (what the old scripts did; kept for continuity, NOT primary)
  trajectory cluster-level analysis: collapse each trajectory to its mean, then do ordinary
             one-sample inference on those means with a t distribution (G-1 df). This is the
             textbook-safe analysis and it is our PRIMARY estimand.
  crve       cluster-robust variance estimation on the round-level data (clustered on trajectory),
             with the standard small-sample corrections: G/(G-1) scaling and t(G-1) critical values.
             Keeps unbalanced-round information while getting the SE right.

Also reports the design effect / effective sample size so a reader can see exactly how much the
naive n was overstating the evidence:  n_eff = n / (1 + (m_bar - 1) * ICC).

No numpy/scipy (the cluster has neither reliably); t quantiles come from an exact regularized
incomplete beta with a Lentz continued fraction, inverted by bisection.
"""
import math
import statistics as st

# --------------------------------------------------------------------------------------
# t distribution (exact, pure stdlib)
# --------------------------------------------------------------------------------------


def _betacf(a, b, x, itmax=300, eps=3e-12):
    """Continued fraction for the incomplete beta function (Numerical Recipes, Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    """P(T <= t) for Student-t with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    p_tail = 0.5 * betainc(df / 2.0, 0.5, x)   # P(|T| > |t|) / 2
    return 1.0 - p_tail if t > 0 else p_tail


def t_ppf(p, df):
    """Inverse CDF (quantile) for Student-t, by bisection on the exact CDF."""
    if df <= 0:
        return float("nan")
    if df > 1e6:
        # normal limit; Acklam-free simple inversion via bisection on erf
        lo, hi = -40.0, 40.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    lo, hi = -1e3, 1e3
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------------------


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def round_level(clusters):
    """Naive pooling of every round as if independent. Reported for continuity only."""
    vals = [x for c in clusters for x in c]
    n = len(vals)
    if n < 2:
        return None
    m, sd = _mean(vals), st.stdev(vals)
    se = sd / math.sqrt(n)
    return {"unit": "round", "n": n, "G": len(clusters), "mean": m, "se": se, "df": n - 1}


def trajectory_level(clusters):
    """PRIMARY. Collapse each trajectory to its mean; one-sample t on the G cluster means."""
    means = [_mean(c) for c in clusters if c]
    G = len(means)
    if G < 2:
        return None
    m, sd = _mean(means), st.stdev(means)
    se = sd / math.sqrt(G)
    return {"unit": "trajectory", "n": G, "G": G, "mean": m, "se": se, "df": G - 1,
            "n_rounds": sum(len(c) for c in clusters)}


def cluster_robust(clusters):
    """Cluster-robust (sandwich) SE for the grand mean, clustered on trajectory.

    For the intercept-only model y_ij = mu + e_ij, the CRVE reduces to
        Var(mu_hat) = (sum_g r_g^2) / N^2,   r_g = sum_i (y_ij - mu_hat),
    with the usual finite-cluster correction G/(G-1) and t(G-1) critical values.
    """
    vals = [x for c in clusters for x in c]
    N, G = len(vals), len([c for c in clusters if c])
    if N < 2 or G < 2:
        return None
    m = _mean(vals)
    ssr = sum(sum(x - m for x in c) ** 2 for c in clusters if c)
    var = (G / (G - 1.0)) * ssr / (N * N)
    return {"unit": "cluster-robust", "n": N, "G": G, "mean": m,
            "se": math.sqrt(var), "df": G - 1}


def icc_design_effect(clusters):
    """One-way-ANOVA ICC and the design effect that follows from it.

    design_effect = 1 + (m_bar - 1) * ICC ;  n_eff = N / design_effect.
    An ICC near 0 would justify round-level pooling; anything appreciable does not.
    """
    cs = [c for c in clusters if c]
    G, N = len(cs), sum(len(c) for c in cs)
    if G < 2 or N <= G:
        return None
    grand = _mean([x for c in cs for x in c])
    msb = sum(len(c) * (_mean(c) - grand) ** 2 for c in cs) / (G - 1)
    msw = sum((x - _mean(c)) ** 2 for c in cs for x in c) / (N - G)
    # m0: average cluster size correction for unbalanced designs
    m0 = (N - sum(len(c) ** 2 for c in cs) / N) / (G - 1)
    icc = 0.0 if (msb + (m0 - 1) * msw) == 0 else (msb - msw) / (msb + (m0 - 1) * msw)
    icc = max(0.0, min(1.0, icc))
    m_bar = N / G
    deff = 1.0 + (m_bar - 1.0) * icc
    return {"icc": icc, "m_bar": m_bar, "design_effect": deff, "n_eff": N / deff if deff else N}


# --------------------------------------------------------------------------------------
# inference on top of an estimator dict
# --------------------------------------------------------------------------------------


def ci(est, level=0.95):
    if not est:
        return None
    q = t_ppf(0.5 + level / 2.0, est["df"])
    return (est["mean"] - q * est["se"], est["mean"] + q * est["se"])


def tost(est, delta):
    """Two one-sided tests. Equivalent iff the (1-2a) CI lies strictly inside [-delta, +delta].
    Uses the t distribution at the estimator's df (NOT a normal z, which is what the old script
    assumed and which is anticonservative at the cluster-level sample sizes we actually have)."""
    if not est:
        return None
    q = t_ppf(0.95, est["df"])            # 90% CI for TOST at alpha=0.05 each side
    lo, hi = est["mean"] - q * est["se"], est["mean"] + q * est["se"]
    t_lower = (est["mean"] + delta) / est["se"]     # H0: mu <= -delta
    t_upper = (delta - est["mean"]) / est["se"]     # H0: mu >= +delta
    p_lower = 1.0 - t_cdf(t_lower, est["df"])
    p_upper = 1.0 - t_cdf(t_upper, est["df"])
    return {"ci90": (lo, hi), "equivalent": (lo > -delta) and (hi < delta),
            "p_tost": max(p_lower, p_upper), "delta": delta}


def bf01(est):
    """BIC approximation to BF01 (Wagenmakers 2007 eq. 11) evaluated at the estimator's OWN n.

    BF01 = sqrt(n) * (1 + t^2/(n-1))^(-n/2).
    Feeding this the cluster count rather than the round count is the whole point: the naive
    version bought its evidence from replicated, correlated rounds.
    """
    if not est or est["se"] == 0:
        return None
    n = est["n"] if est["unit"] != "cluster-robust" else est["G"]
    if n < 2:
        return None
    t = est["mean"] / est["se"]
    return math.sqrt(n) * (1 + t * t / (n - 1)) ** (-n / 2.0)


def analyze(clusters, delta=0.05, level=0.95):
    """Run all three estimands plus ICC on a list-of-lists (one inner list per trajectory)."""
    out = {}
    for name, fn in (("round", round_level), ("trajectory", trajectory_level),
                     ("crve", cluster_robust)):
        est = fn(clusters)
        if not est:
            out[name] = None
            continue
        out[name] = {**est, "ci": ci(est, level), "tost": tost(est, delta), "bf01": bf01(est)}
    out["icc"] = icc_design_effect(clusters)
    return out


def fmt_row(label, a, width=30):
    """One printable line for an analyze() sub-result."""
    if not a:
        return "%-*s  (insufficient data)" % (width, label[:width])
    lo, hi = a["ci"]
    t = a["tost"]
    bf = a["bf01"]
    return "%-*s %4d %4d  %+.4f  [%+.4f,%+.4f]  %-9s  %s" % (
        width, label[:width], a["n"], a["G"], a["mean"], lo, hi,
        ("EQUIV" if t and t["equivalent"] else "not-equiv"),
        ("%6.1f" % bf) if bf is not None and bf < 1e4 else "    --")


HEADER = "%-30s    n    G     mean         95%% CI            TOST      BF01" % ""


if __name__ == "__main__":
    # self-test: a known-correlated design must show a materially smaller effective n
    import random
    rng = random.Random(0)
    clusters = []
    for _ in range(12):                      # 12 trajectories
        base = rng.gauss(0, 0.05)            # trajectory-level offset -> induces ICC
        clusters.append([base + rng.gauss(0, 0.02) for _ in range(6)])
    a = analyze(clusters)
    print(HEADER)
    for k in ("round", "trajectory", "crve"):
        print(fmt_row(k, a[k]))
    print("ICC=%.3f design_effect=%.2f n_eff=%.1f (raw n=%d)" %
          (a["icc"]["icc"], a["icc"]["design_effect"], a["icc"]["n_eff"], a["round"]["n"]))
    assert a["icc"]["icc"] > 0.5, "self-test: strong cluster structure must be detected"
    assert a["icc"]["n_eff"] < 0.4 * a["round"]["n"], "self-test: n_eff must collapse"
    assert a["crve"]["se"] > 2 * a["round"]["se"], "self-test: CRVE must widen the SE"
    assert abs(t_ppf(0.975, 12) - 2.179) < 1e-3, "self-test: t quantile"
    assert abs(t_ppf(0.95, 12) - 1.782) < 1e-3, "self-test: t quantile"
    assert abs(t_cdf(0.0, 5) - 0.5) < 1e-12
    print("PASS")
