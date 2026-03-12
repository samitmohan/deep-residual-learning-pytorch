#!/usr/bin/env bash
# Trains all CIFAR-10 plain and residual network variants from Table 6
# of "Deep Residual Learning for Image Recognition" (He et al., 2015).
#
# Usage: bash run_experiments.sh

set -euo pipefail

DEPTHS=(20 32 44 56)

echo "=== Training plain networks ==="
for n in "${DEPTHS[@]}"; do
    echo "--- Plain-${n} ---"
    uv run train.py "$n"
done

echo "=== Training residual networks (Option A) ==="
for n in "${DEPTHS[@]}"; do
    echo "--- ResNet-${n} Option A ---"
    uv run train.py "$n" -r -o A
done

echo "=== Training residual networks (Option B, depth 20 only) ==="
echo "--- ResNet-20 Option B ---"
uv run train.py 20 -r -o B

echo "=== Generating plots ==="
uv run results_plot.py

echo "=== Done ==="
