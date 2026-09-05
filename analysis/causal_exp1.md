# Causal experiment #1: memory attribution (Fable 5.1, pilot)

Four matched arms, 2 independent campaigns each, Medium band, transfer n=12, 4 rounds.
Mean final transfer score C (0 = eager, 1 = expert):

    growing (recursive)   0.287
    frozen-nonempty       0.270
    offline-built         0.235
    matched-search        0.212

The ordering growing > frozen > offline > search is the direction RSI predicts. But the
gaps are far below the measured ~0.15 unchanged-state noise floor at n=2, so they are not
significant. Honest read: having a transferable library beats pure search slightly, and
recursively growing it does not clearly add beyond ordinary construction. Underpowered, not
conclusive. This motivates the full sweep (all models, more seeds) and, decisively, the
solver-improver transplant (experiment #4).
