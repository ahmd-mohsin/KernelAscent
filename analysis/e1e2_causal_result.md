# E1/E2 causal diagnostics — results

Instrument: `kernelascent/v3/e1e2.py` (prescriptive/constraining NEG) and `e1e2_pos.py`
(capability-additive best-of-3 POS), on the validated core (`core.py`). Deterministic calibration
7/7 (neg) and 6/6 (pos) PASS. 8 lineage-paired blocks/model, anchor-n=3, held-out seeds. 95% CIs.

## Headline (capability-stratified)
- USEFULNESS dQ: additive raises Q for all (Coder-7B +0.167*, gpt-oss +0.083*, Fable +0.104*);
  prescriptive lowers Q (Coder-7B -0.167*, Fable -0.542*). Sign flips with additive-vs-constraining;
  constraining hurts stronger models more. (* CI excludes 0.)
- CAUSAL SELF-USE F_selfuse: resolved-positive ONLY at the frontier (Fable +0.104 [0.018,0.190]);
  Coder-7B 0, gpt-oss spans 0.
- SELF-REVISION N_selfuse: significantly negative for sub-frontier (Coder-7B -0.583*, gpt-oss
  -0.354*) — self-modification degrades; neutral at frontier (Fable +0.062, spans 0).
- Interpretation: a FINITE causal self-use link at the frontier, NOT compounding RSI (one link;
  F_channel spans 0; coarse 3-level score bins children -> F often exactly 0). Follow-ups: speed-
  resolved score (E-wall), repeat-link, rescue, replication at the frontier.

## Raw aggregates
```
## NEG (prescriptive)
### coder7b
{
  "who": "hf:Qwen/Qwen2.5-Coder-7B-Instruct",
  "blocks_target": 8,
  "blocks_done": 8,
  "E1": {
    "dQ_useful": {
      "mean": -0.1667,
      "n": 8,
      "sd": 0.1992,
      "ci95": [
        -0.3047,
        -0.0286
      ]
    },
    "F_channel": {
      "mean": 0.0833,
      "n": 8,
      "sd": 0.0891,
      "ci95": [
        0.0216,
        0.1451
      ]
    },
    "dQ_null": {
      "mean": -0.2708,
      "n": 8,
      "sd": 0.1527,
      "ci95": [
        -0.3766,
        -0.165
      ]
    },
    "F_null": {
      "mean": 0.0417,
      "n": 8,
      "sd": 0.2315,
      "ci95": [
        -0.1187,
        0.2021
      ]
    }
  },
  "E2": {
    "F_selfuse": {
      "mean": -0.1042,
      "n": 8,
      "sd": 0.198,
      "ci95": [
        -0.2413,
        0.033
      ]
    },
    "N_selfuse": {
      "mean": 0.0625,
      "n": 8,
      "sd": 0.1768,
      "ci95": [
        -0.06,
        0.185
      ]
    },
    "rescue_minus_revert": {
      "mean": 0.0833,
      "n": 8,
      "sd": 0.126,
      "ci95": [
        -0.004,
        0.1706
      ]
    }
  },
  "decompose": {
    "n_dev": 456,
    "correct_rate": 0.526,
    "fast_rate": 0.007
  }
}
### fable
{
  "who": "api:us.anthropic.claude-fable-5-1",
  "blocks_target": 8,
  "blocks_done": 8,
  "E1": {
    "dQ_useful": {
      "mean": -0.5417,
      "n": 8,
      "sd": 0.3054,
      "ci95": [
        -0.7533,
        -0.3301
      ]
    },
    "F_channel": {
      "mean": 0.0625,
      "n": 8,
      "sd": 0.0863,
      "ci95": [
        0.0027,
        0.1223
      ]
    },
    "dQ_null": {
      "mean": -0.625,
      "n": 8,
      "sd": 0.1179,
      "ci95": [
        -0.7067,
        -0.5433
      ]
    },
    "F_null": {
      "mean": -0.0833,
      "n": 8,
      "sd": 0.2182,
      "ci95": [
        -0.2346,
        0.0679
      ]
    }
  },
  "E2": {
    "F_selfuse": {
      "mean": -0.0833,
      "n": 8,
      "sd": 0.126,
      "ci95": [
        -0.1706,
        0.004
      ]
    },
    "N_selfuse": {
      "mean": -0.0208,
      "n": 8,
      "sd": 0.1068,
      "ci95": [
        -0.0949,
        0.0532
      ]
    },
    "rescue_minus_revert": {
      "mean": -0.0417,
      "n": 8,
      "sd": 0.1179,
      "ci95": [
        -0.1233,
        0.04
      ]
    }
  },
  "decompose": {
    "n_dev": 456,
    "correct_rate": 0.897,
    "fast_rate": 0.254
  }
}
## POS (capability-additive best-of-3)
### coder7b
{
  "who": "hf:Qwen/Qwen2.5-Coder-7B-Instruct",
  "budget": 3,
  "blocks_target": 8,
  "blocks_done": 8,
  "E1": {
    "dQ_useful": {
      "mean": 0.1667,
      "n": 8,
      "sd": 0.1782,
      "ci95": [
        0.0432,
        0.2901
      ]
    },
    "F_channel": {
      "mean": 0.0,
      "n": 8,
      "sd": 0.0,
      "ci95": [
        0.0,
        0.0
      ]
    },
    "dQ_null": {
      "mean": 0.1458,
      "n": 8,
      "sd": 0.243,
      "ci95": [
        -0.0225,
        0.3142
      ]
    }
  },
  "E2": {
    "F_selfuse": {
      "mean": 0.0,
      "n": 8,
      "sd": 0.0,
      "ci95": [
        0.0,
        0.0
      ]
    },
    "N_selfuse": {
      "mean": -0.5833,
      "n": 8,
      "sd": 0.0891,
      "ci95": [
        -0.6451,
        -0.5216
      ]
    },
    "rescue_minus_revert": {
      "mean": 0.0,
      "n": 8,
      "sd": 0.0,
      "ci95": [
        0.0,
        0.0
      ]
    }
  },
  "decompose": {
    "n_dev": 568,
    "correct_rate": 0.222,
    "fast_rate": 0.025
  }
}
### gptoss
{
  "who": "api:openai.gpt-oss-120b-1:0",
  "budget": 3,
  "blocks_target": 8,
  "blocks_done": 8,
  "E1": {
    "dQ_useful": {
      "mean": 0.0833,
      "n": 8,
      "sd": 0.0891,
      "ci95": [
        0.0216,
        0.1451
      ]
    },
    "F_channel": {
      "mean": 0.0,
      "n": 8,
      "sd": 0.0,
      "ci95": [
        0.0,
        0.0
      ]
    },
    "dQ_null": {
      "mean": 0.1042,
      "n": 8,
      "sd": 0.1768,
      "ci95": [
        -0.0183,
        0.2267
      ]
    }
  },
  "E2": {
    "F_selfuse": {
      "mean": 0.125,
      "n": 8,
      "sd": 0.2315,
      "ci95": [
        -0.0354,
        0.2854
      ]
    },
    "N_selfuse": {
      "mean": -0.3542,
      "n": 8,
      "sd": 0.3825,
      "ci95": [
        -0.6192,
        -0.0891
      ]
    },
    "rescue_minus_revert": {
      "mean": 0.0625,
      "n": 8,
      "sd": 0.1768,
      "ci95": [
        -0.06,
        0.185
      ]
    }
  },
  "decompose": {
    "n_dev": 568,
    "correct_rate": 0.342,
    "fast_rate": 0.093
  }
}
### fable
{
  "who": "api:us.anthropic.claude-fable-5-1",
  "budget": 3,
  "blocks_target": 8,
  "blocks_done": 8,
  "E1": {
    "dQ_useful": {
      "mean": 0.1042,
      "n": 8,
      "sd": 0.124,
      "ci95": [
        0.0182,
        0.1901
      ]
    },
    "F_channel": {
      "mean": 0.1042,
      "n": 8,
      "sd": 0.198,
      "ci95": [
        -0.033,
        0.2413
      ]
    },
    "dQ_null": {
      "mean": 0.0833,
      "n": 8,
      "sd": 0.126,
      "ci95": [
        -0.004,
        0.1706
      ]
    }
  },
  "E2": {
    "F_selfuse": {
      "mean": 0.1042,
      "n": 8,
      "sd": 0.124,
      "ci95": [
        0.0182,
        0.1901
      ]
    },
    "N_selfuse": {
      "mean": 0.0625,
      "n": 8,
      "sd": 0.1768,
      "ci95": [
        -0.06,
        0.185
      ]
    },
    "rescue_minus_revert": {
      "mean": 0.1042,
      "n": 8,
      "sd": 0.2346,
      "ci95": [
        -0.0584,
        0.2668
      ]
    }
  },
  "decompose": {
    "n_dev": 568,
    "correct_rate": 0.201,
    "fast_rate": 0.121
  }
}

```
