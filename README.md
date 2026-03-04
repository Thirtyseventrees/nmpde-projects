## Project 2: 2D Wave Equation (FEM)

This project implements a 2D finite element solver for the wave equation using **deal.II** on simplex meshes:

$$
u_{tt} - \Delta u = f \quad \text{in } \Omega,
\qquad u = g \quad \text{on } \partial\Omega.
$$

The current implementation uses homogeneous Dirichlet boundary conditions (`g=0`), default source-free dynamics (`f=0`), and a fixed **eigenmode** initial condition.

## Implemented Discretization Choices

The solver now supports multiple numerical choices:

1. Time discretization:
- `cd`: explicit central difference (second order in time)
- `newmark`: implicit Newmark method with `beta=0.25`, `gamma=0.5`

2. Mass treatment:
- `lumped`: row-sum lumped mass (diagonal)
- `consistent`: full consistent mass matrix

3. Space discretization:
- `FE_SimplexP<2>(k)` with configurable polynomial degree `k` (`fe_degree >= 1`)

## Code Structure

- [CMakeLists.txt](/home/chenyuye/homework/pde_project/CMakeLists.txt): build configuration
- [src/main.cpp](/home/chenyuye/homework/pde_project/src/main.cpp): CLI parsing, solver setup, run entry
- [src/Wave2D.hpp](/home/chenyuye/homework/pde_project/src/Wave2D.hpp): solver class, enums, and interfaces
- [src/Wave2D.cpp](/home/chenyuye/homework/pde_project/src/Wave2D.cpp): matrix assembly, boundary handling, time stepping, output, error/energy evaluation
- `scripts/mesh-square.geo`: Gmsh geometry for unit square meshes
- `scripts/generate-meshes.sh`: batch mesh generation
- `scripts/plot_energy.py`: energy and relative drift plotting from CSV
- `scripts/plot_error.py`: L2/H1 error plotting from CSV
- `scripts/plot_3d_surface.py`: 3D surface visualization from VTU files
- `scripts/generate_plots.sh`: post-run plotting pipeline
- `scripts/run_experiments.sh`: batch runner for all convergence/dissipation/dispersion experiments
- `scripts/run_report_supplement.sh`: batch runner for extra report figures (Newmark parameter, boundary driving, CFL scan)
- `scripts/plot_convergence.py`: h- and dt-convergence rate plots (log-log with fitted slopes)
- `scripts/plot_energy_comparison.py`: overlay energy curves across schemes (dissipation analysis)
- `scripts/plot_error_comparison.py`: overlay error curves across schemes (dispersion analysis)
- `scripts/plot_dispersion_from_results.py`: data-driven dispersion relation and amplification factor plots from solver outputs
- `scripts/run_dispersion_experiments.sh`: mode-sweep batch runs used to build data-driven dispersion/dissipation figures
- `scripts/dispersion_analysis.py`: theoretical dispersion relation and amplification factor plots (reference only)
- `scripts/plot_report_supplement.py`: generates `supplement_*.png` figures used to complete report analysis

## Build

Example (course/module environment):

```bash
cd /home/chenyuye/homework/pde_project
module load gcc-glibc dealii
mkdir -p build
cd build
cmake ..
make -j
```

Clean all run outputs:

```bash
cd /home/chenyuye/homework/pde_project/build
make clean-results
```

## Run

```bash
cd /home/chenyuye/homework/pde_project/build
./main [mesh.msh] [dt] [T] [output_every] [omega] [time_scheme] [mass_type] [fe_degree] [compute_error_each_step] [auto_plot]
       [boundary_mode] [newmark_beta] [newmark_gamma] [mode_x] [mode_y]
```

Arguments:

1. `mesh.msh`: Gmsh mesh file path
2. `dt`: time step size
3. `T`: final time
4. `output_every`: output VTU every N steps (no output when it equal to 0)
5. `omega`: angular frequency (used for `eigenmode` exact solution)
6. `time_scheme`: `cd` (default) or `newmark`
7. `mass_type`: `lumped` (default) or `consistent`
8. `fe_degree`: polynomial degree (default `1`)
9. `compute_error_each_step`: `1` (default) to compute/write error every step, `0` to write only final error
10. `auto_plot`: `1` (default) to auto-run plotting scripts, `0` to skip auto plotting
11. `boundary_mode`: `homogeneous` (default) or `driven` (time-dependent boundary forcing)
12. `newmark_beta`: Newmark parameter `beta` (default `0.25`, used when `time_scheme=newmark`)
13. `newmark_gamma`: Newmark parameter `gamma` (default `0.5`, used when `time_scheme=newmark`)
14. `mode_x`: eigenmode index in `x` for manufactured solution (default `1`)
15. `mode_y`: eigenmode index in `y` for manufactured solution (default `1`)

Default-equivalent run:

```bash
./main ../mesh/mesh-square-h0.1.msh 0.01 2.0 1 4.442882938 cd lumped 1 1 1
```

## Output Files

All outputs are written under:

- `result/<run_config>/` (at the project root)

where `<run_config>` is automatically generated from input parameters:
mesh, time scheme, mass type, FE degree, `dt`, `T`, `output_every`, and `omega`.

Inside each run folder, outputs are tagged by method:

- `output-<mesh>-<method_tag>-<step>.vtu`
- `energy-<method_tag>.csv`
- `error-<method_tag>.csv`
- `probe-<method_tag>.csv` (per-step probe values for data-driven dispersion/dissipation diagnostics)
- `run-meta.csv` (run metadata: `h_min`, `dt`, `cfl`, mode indices, etc.)
- `energy-<method_tag>.png` (auto-generated after run)
- `error-<method_tag>.png` (auto-generated after run)

where `method_tag = <time_scheme>-<mass_type>-p<fe_degree>`.

`error-<method_tag>.csv` columns:
- `step,time,L2_error,H1_error`

## Analysis Workflow

### 1. Run all experiments

```bash
bash scripts/run_experiments.sh
```

This runs all convergence studies (h and dt), scheme comparisons, and FE degree comparisons, then generates all comparison plots automatically.

### 2. Data-driven dispersion / dissipation diagnostics

```bash
bash scripts/run_dispersion_experiments.sh
# or (if runs already exist)
python3 scripts/plot_dispersion_from_results.py result/ --h 0.05 --p 1 --show
```

Produces `dispersion_relation.png` (phase velocity error),
`dissipation_amplification_central.png`, and
`dissipation_amplification_newmark.png` (amplitude ratio per step).

### 3. Theoretical reference curves (optional)

```bash
python3 scripts/dispersion_analysis.py result/ --show
```

### 4. Individual comparison plots (after solver runs)

```bash
python3 scripts/plot_convergence.py result/
python3 scripts/plot_energy_comparison.py result/      # dissipation
python3 scripts/plot_error_comparison.py result/        # dispersion

```

```bash
bash scripts/run_report_supplement.sh
```

This additionally generates:
- `result/supplement_newmark_dissipation.png`
- `result/supplement_boundary_energy.png`
- `result/supplement_cfl_scan.png`
