#!/bin/bash
# CI self-check: the benchmark's deterministic gates must pass (no model/GPU/creds).
set -e
python3 -u kernelascent/v3/rsi_verify.py --gate2
python3 -u kernelascent/v3/rsi_verify.py --calib
echo "SELF-CHECK OK"
