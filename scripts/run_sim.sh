#!/usr/bin/env bash
set -euo pipefail

# Quick convenience wrapper for local simulation runs.
# Example:
#   ./scripts/run_sim.sh --algo fedavg --task vision/cifar10 --model cnn/moderate \
#     --n_clients 8 --num_rounds 20 --alpha 0.5

flbench-run "$@"
