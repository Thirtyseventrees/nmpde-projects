#!/bin/bash
# Generate multiple meshes for h-convergence study.
# Usage: bash generate-meshes.sh
# Requires: gmsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../mesh"
mkdir -p "$OUT_DIR"

for h in 0.2 0.1 0.05 0.025; do
  echo "Generating mesh with h=$h ..."
  gmsh "$SCRIPT_DIR/mesh-square.geo" -setnumber h "$h" -2 -format msh4 -o "$OUT_DIR/mesh-square-h${h}.msh"
done

echo "Done. Meshes saved in $OUT_DIR/"
ls -la "$OUT_DIR/"
