#!/usr/bin/env bash
set -u

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/generate_plots.sh <run_output_dir>"
  exit 1
fi

RUN_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[warning] run directory not found: $RUN_DIR"
  exit 0
fi

PYTHON_BIN="python3"
if [[ -x "$PROJECT_DIR/.venv/bin/python3" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
fi

plot_with_script() {
  local script_name="$1"
  local csv_file="$2"

  if [[ ! -f "$SCRIPT_DIR/$script_name" ]]; then
    echo "[warning] missing script: $SCRIPT_DIR/$script_name"
    return 0
  fi

  if [[ ! -f "$csv_file" ]]; then
    return 0
  fi

  MPLBACKEND=Agg "$PYTHON_BIN" "$SCRIPT_DIR/$script_name" "$csv_file" || \
    echo "[warning] plotting failed: $script_name $csv_file"
}

shopt -s nullglob
for csv in "$RUN_DIR"/energy-*.csv; do
  plot_with_script "plot_energy.py" "$csv"
done

for csv in "$RUN_DIR"/error-*.csv; do
  plot_with_script "plot_error.py" "$csv"
done
