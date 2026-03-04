#!/usr/bin/env bash
# ============================================================
# Batch runner for wave equation experiments.
#
# Runs the solver with multiple configurations for:
#   1. h-convergence study
#   2. dt-convergence study
#   3. Scheme comparison (dissipation / dispersion)
#   4. FE degree comparison
#
# Usage:
#   bash scripts/run_experiments.sh            # from project root
#   bash scripts/run_experiments.sh /path/to/build/main
#
# Prerequisites: build the project first (cmake .. && make -j)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

MAIN="${1:-$BUILD_DIR/main}"

if [[ ! -x "$MAIN" ]]; then
  echo "[error] executable not found: $MAIN"
  echo "        Build the project first:  cd build && cmake .. && make -j"
  exit 1
fi

MESH_DIR="$PROJECT_DIR/mesh"
OMEGA="4.442882938"  # sqrt(2)*pi for eigenmode sin(pi*x)*sin(pi*y)*cos(omega*t)
T="2.0"
# Current main signature:
#   ./main mesh dt T output_every omega time_scheme mass_type fe_degree compute_error_each_step auto_plot
AUTO_PLOT=0  # run comparison plots once at the end
COMPUTE_ERROR_EACH_STEP=1

echo "========================================"
echo "  Wave equation batch experiments"
echo "========================================"

run_case() {
  local mesh="$1" dt="$2" out_every="$3" scheme="$4" mass="$5" p="$6"
  echo ""
  echo "--- Running: mesh=$(basename $mesh) dt=$dt scheme=$scheme mass=$mass p=$p ---"
  "$MAIN" "$mesh" "$dt" "$T" "$out_every" "$OMEGA" \
    "$scheme" "$mass" "$p" "$COMPUTE_ERROR_EACH_STEP" "$AUTO_PLOT"
}

# ============================================================
#  1. h-convergence  (fix dt small, vary h)
# ============================================================
echo ""
echo "========================================="
echo "  Experiment 1: h-convergence (cd+lumped P1)"
echo "========================================="
DT_FINE="0.002"
for h in 0.2 0.1 0.05 0.025; do
  run_case "$MESH_DIR/mesh-square-h${h}.msh" "$DT_FINE" 0 cd lumped 1
done

echo ""
echo "========================================="
echo "  Experiment 1b: h-convergence (cd+lumped P2)"
echo "========================================="
for h in 0.2 0.1 0.05 0.025; do
  run_case "$MESH_DIR/mesh-square-h${h}.msh" "$DT_FINE" 0 cd lumped 2
done

# ============================================================
#  2. dt-convergence  (fix fine mesh, vary dt)
# ============================================================
echo ""
echo "========================================="
echo "  Experiment 2: dt-convergence (cd+lumped P1, h=0.025)"
echo "========================================="
FINE_MESH="$MESH_DIR/mesh-square-h0.025.msh"
for dt_val in 0.04 0.02 0.01 0.005 0.002; do
  run_case "$FINE_MESH" "$dt_val" 0 cd lumped 1
done

echo ""
echo "========================================="
echo "  Experiment 2b: dt-convergence (newmark+consistent P1, h=0.025)"
echo "========================================="
for dt_val in 0.04 0.02 0.01 0.005 0.002; do
  run_case "$FINE_MESH" "$dt_val" 0 newmark consistent 1
done

# ============================================================
#  3. Scheme / mass comparison  (fix mesh & dt, vary scheme+mass)
# ============================================================
echo ""
echo "========================================="
echo "  Experiment 3: Scheme comparison (h=0.05, dt=0.005)"
echo "========================================="
REF_MESH="$MESH_DIR/mesh-square-h0.05.msh"
REF_DT="0.005"
run_case "$REF_MESH" "$REF_DT" 0 cd       lumped     1
run_case "$REF_MESH" "$REF_DT" 0 cd       consistent 1
run_case "$REF_MESH" "$REF_DT" 0 newmark  lumped     1
run_case "$REF_MESH" "$REF_DT" 0 newmark  consistent 1

# ============================================================
#  4. FE degree comparison  (P1 vs P2)
# ============================================================
echo ""
echo "========================================="
echo "  Experiment 4: FE degree comparison (h=0.05, dt=0.005)"
echo "========================================="
run_case "$REF_MESH" "$REF_DT" 0 cd lumped 1
run_case "$REF_MESH" "$REF_DT" 0 cd lumped 2

echo ""
echo "========================================="
echo "  All experiments completed."
echo "========================================="

# ============================================================
#  5. Run comparison plots
# ============================================================
echo ""
echo "Running comparison plotting scripts …"

PYTHON_BIN="python3"
if [[ -x "$PROJECT_DIR/.venv/bin/python3" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
fi

RESULT_DIR="$PROJECT_DIR/result"

# Generate individual energy/error plots for every run
for dir in "$RESULT_DIR"/*/; do
  bash "$SCRIPT_DIR/generate_plots.sh" "$dir" 2>/dev/null || true
done

# Convergence plots
MPLBACKEND=Agg "$PYTHON_BIN" "$SCRIPT_DIR/plot_convergence.py" "$RESULT_DIR" 2>/dev/null || \
  echo "[info] plot_convergence.py skipped or failed"

# Energy comparison (dissipation)
MPLBACKEND=Agg "$PYTHON_BIN" "$SCRIPT_DIR/plot_energy_comparison.py" "$RESULT_DIR" 2>/dev/null || \
  echo "[info] plot_energy_comparison.py skipped or failed"

# Error comparison (dispersion)
MPLBACKEND=Agg "$PYTHON_BIN" "$SCRIPT_DIR/plot_error_comparison.py" "$RESULT_DIR" 2>/dev/null || \
  echo "[info] plot_error_comparison.py skipped or failed"

# Dispersion relation
MPLBACKEND=Agg "$PYTHON_BIN" "$SCRIPT_DIR/dispersion_analysis.py" "$RESULT_DIR" 2>/dev/null || \
  echo "[info] dispersion_analysis.py skipped or failed"

echo ""
echo "Done.  All figures saved under $RESULT_DIR/"
