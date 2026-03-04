#ifndef WAVE_2D_HPP
#define WAVE_2D_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace dealii;

class Wave2D
{
public:
  static constexpr unsigned int dim = 2;

  enum class TimeScheme
  {
    central_difference,
    newmark
  };

  enum class MassType
  {
    lumped,
    consistent
  };

  enum class BoundaryMode
  {
    homogeneous,
    driven
  };

  struct SolverOptions
  {
    TimeScheme   time_scheme = TimeScheme::central_difference;
    MassType     mass_type   = MassType::lumped;
    BoundaryMode boundary_mode = BoundaryMode::homogeneous;
    unsigned int fe_degree   = 1;
    unsigned int mode_x      = 1;
    unsigned int mode_y      = 1;
    double       newmark_beta  = 0.25;
    double       newmark_gamma = 0.5;
  };

  Wave2D(const std::string &mesh_file_name_);

  Wave2D(const std::string   &mesh_file_name_,
         const SolverOptions &options_);

  void
  run(const double       dt,
      const double       T,
      const unsigned int output_every,
      const double       omega,
      const bool         compute_error_each_step = true);

  const std::filesystem::path &
  output_directory() const;

protected:
  void
  setup();

  void
  assemble_matrices();

  void
  compute_lumped_mass();

  void
  apply_dirichlet(Vector<double> &u, const double time) const;

  void
  output(const Vector<double> &u,
         const unsigned int    step,
         const double          time) const;

  double
  compute_energy(const Vector<double> &u, const Vector<double> &v) const;

  double
  compute_L2_error(const Vector<double> &u_h,
                   const Function<dim>  &u_exact) const;

  double
  compute_H1_error(const Vector<double> &u_h,
                   const Function<dim>  &u_exact) const;

  double
  compute_min_cell_diameter() const;

  double
  sample_solution_at_point(const Vector<double> &u,
                           const Point<dim>     &p) const;

  void
  enforce_homogeneous_dirichlet_rows(SparseMatrix<double> &matrix) const;

  void
  apply_inverse_mass(Vector<double>       &dst,
                     const Vector<double> &rhs) const;

  std::string
  method_tag() const;

  const std::string   mesh_file_name;
  const SolverOptions options;

  Triangulation<dim> mesh;
  DoFHandler<dim>    dof_handler;

  std::unique_ptr<FE_SimplexP<dim>>   fe;
  std::unique_ptr<QGaussSimplex<dim>> quadrature;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> stiffness_matrix;
  SparseMatrix<double> mass_matrix_dirichlet;

  Vector<double> mass_lumped;
  IndexSet       boundary_dofs;

  mutable SparseDirectUMFPACK mass_solver;
  bool                        mass_solver_initialized = false;
  std::filesystem::path       run_output_dir;
  std::vector<Point<dim>>     support_points;
};

#endif
