#!/usr/bin/env bash
# Run mode-sweep experiments and generate data-driven dispersion/dissipation plots.
#
# Usage:
#   bash scripts/run_dispersion_experiments.sh
#   bash scripts/run_dispersion_experiments.sh /path/to/build/main
#
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

MESH="${MESH:-$PROJECT_DIR/mesh/mesh-square-h0.05.msh}"
P="${P:-1}"
MODE_MAX="${MODE_MAX:-14}"
MODE_Y="${MODE_Y:-1}"
T_FINAL="${T_FINAL:-4.0}"
SEMI_CFL="${SEMI_CFL:-0.1}"
CFL_LIST="${CFL_LIST:-0.1 0.5 0.8 1.0}"
RESULT_DIR="$PROJECT_DIR/result"

if [[ ! -f "$MESH" ]]; then
  echo "[error] mesh not found: $MESH"
  exit 1
fi

# Infer nominal h from mesh filename: ...h0.05.msh
H_VAL="$(basename "$MESH" | sed -E 's/.*h([0-9.]+)\.msh/\1/')"
if [[ -z "$H_VAL" || "$H_VAL" == "$(basename "$MESH")" ]]; then
  echo "[error] cannot infer h from mesh filename: $MESH"
  echo "        Use a mesh named like mesh-square-h0.05.msh"
  exit 1
fi

H_MIN="$(awk -v h="$H_VAL" 'BEGIN{printf "%.12f", sqrt(2.0)*h}')"

run_case() {
  local scheme="$1" mass="$2" cfl="$3" mx="$4" my="$5"
  local dt omega
  dt="$(awk -v hm="$H_MIN" -v cfl="$cfl" 'BEGIN{printf "%.12f", hm*cfl}')"
  omega="$(awk -v mx="$mx" -v my="$my" 'BEGIN{pi=3.14159265358979323846; printf "%.12f", pi*sqrt(mx*mx+my*my)}')"

  "$MAIN" "$MESH" "$dt" "$T_FINAL" 0 "$omega" \
    "$scheme" "$mass" "$P" 0 0 homogeneous 0.25 0.5 "$mx" "$my" >/dev/null
}

echo "=========================================="
echo "  Dispersion mode-sweep experiments"
echo "=========================================="
echo "mesh=$MESH (h=$H_VAL, h_min~$H_MIN), p=$P, modes=1..$MODE_MAX"

# Semi-discrete approximation via very small CFL (Newmark only).
for mass in lumped consistent; do
  for ((mx=1; mx<=MODE_MAX; ++mx)); do
    echo "[semi] newmark/$mass cfl=$SEMI_CFL mode=($mx,$MODE_Y)"
    run_case newmark "$mass" "$SEMI_CFL" "$mx" "$MODE_Y"
  done
done

# Fully discrete curves for both schemes at target CFL values.
for scheme in cd newmark; do
  for mass in lumped consistent; do
    for cfl in $CFL_LIST; do
      for ((mx=1; mx<=MODE_MAX; ++mx)); do
        echo "[$scheme/$mass] cfl=$cfl mode=($mx,$MODE_Y)"
        run_case "$scheme" "$mass" "$cfl" "$mx" "$MODE_Y"
      done
    done
  done
done

PYTHON_BIN="python3"
if [[ -x "$PROJECT_DIR/.venv/bin/python3" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
fi

echo ""
echo "Generating data-driven dispersion/dissipation figures ..."
MPLBACKEND=Agg "$PYTHON_BIN" "$SCRIPT_DIR/plot_dispersion_from_results.py" \
  "$RESULT_DIR" --h "$H_VAL" --p "$P" \
  --cfl-values "0.5,0.8,1.0" \
  --cd-cfl-values "0.1" \
  --amp-cfl-values "0.1" \
  --amp-mass-types "lumped" \
  --semi-cfl "$SEMI_CFL" \
  --min-kh "1.0"

echo "Done."
echo "  $RESULT_DIR/dispersion_relation.png"
echo "  $RESULT_DIR/dissipation_amplification_central.png"
echo "  $RESULT_DIR/dissipation_amplification_newmark.png"
