#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
MAIN="${1:-$BUILD_DIR/main}"

if [[ ! -x "$MAIN" ]]; then
  echo "[error] executable not found: $MAIN"
  echo "        Build first: cd build && cmake .. && make -j"
  exit 1
fi

MESH_DIR="$PROJECT_DIR/mesh"
RESULT_DIR="$PROJECT_DIR/result"
OMEGA="4.442882938"
T="2.0"

run_case() {
  local mesh="$1" dt="$2" out_every="$3" scheme="$4" mass="$5" p="$6"
  local err_each="$7" auto_plot="$8" bc="$9" nmb="${10}" nmg="${11}"
  echo ""
  echo "--- Running: $(basename "$mesh") dt=$dt $scheme/$mass p=$p bc=$bc beta=$nmb gamma=$nmg ---"
  "$MAIN" "$mesh" "$dt" "$T" "$out_every" "$OMEGA" \
    "$scheme" "$mass" "$p" "$err_each" "$auto_plot" "$bc" "$nmb" "$nmg"
}

echo "========================================"
echo "  Supplement experiments for report"
echo "========================================"

# A) Newmark parameter comparison: conservative vs dissipative
MESH_REF="$MESH_DIR/mesh-square-h0.05.msh"
DT_REF="0.005"
run_case "$MESH_REF" "$DT_REF" 0 newmark consistent 1 0 0 homogeneous 0.25 0.5
run_case "$MESH_REF" "$DT_REF" 0 newmark consistent 1 0 0 homogeneous 0.30 0.6

# B) Boundary driving comparison with the conservative Newmark choice
run_case "$MESH_REF" "$DT_REF" 0 newmark consistent 1 0 0 homogeneous 0.25 0.5
run_case "$MESH_REF" "$DT_REF" 0 newmark consistent 1 0 0 driven      0.25 0.5

# C) CFL sweep for explicit central difference (fixed h)
MESH_FINE="$MESH_DIR/mesh-square-h0.025.msh"
for dt in 0.002 0.005 0.01 0.02 0.04; do
  run_case "$MESH_FINE" "$dt" 0 cd lumped 1 1 0 homogeneous 0.25 0.5
done

PYTHON_BIN="python3"
if [[ -x "$PROJECT_DIR/.venv/bin/python3" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
fi

echo ""
echo "Generating supplement figures ..."
MPLBACKEND=Agg "$PYTHON_BIN" "$SCRIPT_DIR/plot_report_supplement.py" "$RESULT_DIR"

echo ""
echo "Done. Supplement figures saved in $RESULT_DIR"
