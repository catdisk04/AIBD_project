#!/usr/bin/env bash
# Reproduce every table and figure in results/ (about five minutes on a laptop; 06 is the slow one).
set -euo pipefail
cd "$(dirname "$0")/code"
for s in 01_batch_correction 02_gex_classification 03_snp_analysis 04_differential_expression 05_figures 06_sanity_checks; do
  echo "== $s"
  python "$s.py"
done
