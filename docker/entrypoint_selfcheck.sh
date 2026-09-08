#!/bin/bash
# CI self-check: the benchmark's deterministic gates must pass (no model / GPU / creds).
set -e
python3 -u kernelascent/v3/rsi_verify.py --gate2      # opportunity proof
python3 -u kernelascent/v3/rsi_verify.py --calib      # verifier causal instrument
python3 -u kernelascent/v3/rsi_true.py   --calib      # verifier-improves-verifier instrument
python3 -u kernelascent/v3/lab_easy.py   --calib --lineages 20 --outdir /tmp/selfcheck  # efficiency lab
python3 -c "from kernelascent import evaluate; print('evaluate entrypoint import OK')"
echo "SELF-CHECK OK"
