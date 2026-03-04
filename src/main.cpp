#include "Wave2D.hpp"

#include <cstdlib>
#include <sstream>

int
main(int argc, char **argv)
{
  try
    {
      // Usage:
      //   ./main [mesh.msh] [dt] [T] [output_every] [omega]
      //          [time_scheme] [mass_type] [fe_degree]
      //          [compute_error_each_step] [auto_plot]
      //          [boundary_mode] [newmark_beta] [newmark_gamma]
      //          [mode_x] [mode_y]
      //
      // time_scheme: "cd" (default) or "newmark"
      // mass_type: "lumped" (default) or "consistent"
      // compute_error_each_step: 1 (default) or 0
      // auto_plot: 1 (default) or 0
      // boundary_mode: "homogeneous" (default) or "driven"
      const std::string mesh_file =
        (argc > 1 ? argv[1] : std::string("../../mesh-square-h0.100000.msh"));
      const double       dt           = (argc > 2 ? std::stod(argv[2]) : 0.01);
      const double       T            = (argc > 3 ? std::stod(argv[3]) : 2.0);
      const unsigned int output_every = (argc > 4 ? std::stoul(argv[4]) : 10);
      const double       omega =
        (argc > 5 ? std::stod(argv[5]) : std::sqrt(2.0) * numbers::PI);
      const std::string time_scheme_str = (argc > 6 ? argv[6] : "cd");
      const std::string mass_type_str   = (argc > 7 ? argv[7] : "lumped");
      const unsigned int fe_degree =
        (argc > 8 ? std::stoul(argv[8]) : 1u);
      const bool compute_error_each_step =
        (argc > 9 ? std::stoul(argv[9]) != 0u : true);
      const bool auto_plot = (argc > 10 ? std::stoul(argv[10]) != 0u : true);
      const std::string boundary_mode_str =
        (argc > 11 ? argv[11] : "homogeneous");
      const double newmark_beta  = (argc > 12 ? std::stod(argv[12]) : 0.25);
      const double newmark_gamma = (argc > 13 ? std::stod(argv[13]) : 0.5);
      const unsigned int mode_x  = (argc > 14 ? std::stoul(argv[14]) : 1u);
      const unsigned int mode_y  = (argc > 15 ? std::stoul(argv[15]) : 1u);

      Wave2D::SolverOptions options;
      if (time_scheme_str == "cd")
        options.time_scheme = Wave2D::TimeScheme::central_difference;
      else if (time_scheme_str == "newmark")
        options.time_scheme = Wave2D::TimeScheme::newmark;
      else
        throw std::runtime_error("Unknown time_scheme: " + time_scheme_str);

      if (mass_type_str == "lumped")
        options.mass_type = Wave2D::MassType::lumped;
      else if (mass_type_str == "consistent")
        options.mass_type = Wave2D::MassType::consistent;
      else
        throw std::runtime_error("Unknown mass_type: " + mass_type_str);

      if (boundary_mode_str == "homogeneous")
        options.boundary_mode = Wave2D::BoundaryMode::homogeneous;
      else if (boundary_mode_str == "driven")
        options.boundary_mode = Wave2D::BoundaryMode::driven;
      else
        throw std::runtime_error("Unknown boundary_mode: " + boundary_mode_str);

      if (fe_degree < 1)
        throw std::runtime_error("fe_degree must be >= 1.");
      if (mode_x < 1 || mode_y < 1)
        throw std::runtime_error("mode_x and mode_y must be >= 1.");
      options.fe_degree = fe_degree;
      options.mode_x    = mode_x;
      options.mode_y    = mode_y;
      options.newmark_beta  = newmark_beta;
      options.newmark_gamma = newmark_gamma;
      if (options.time_scheme == Wave2D::TimeScheme::newmark)
        {
          if (options.newmark_beta <= 0.0)
            throw std::runtime_error("newmark_beta must be > 0.");
          if (options.newmark_gamma <= 0.0)
            throw std::runtime_error("newmark_gamma must be > 0.");
        }

      Wave2D problem(mesh_file, options);
      problem.run(dt, T, output_every, omega, compute_error_each_step);

      if (auto_plot)
        {
          const std::filesystem::path plot_script =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "scripts" /
            "generate_plots.sh";

          if (std::filesystem::exists(plot_script))
            {
              const std::filesystem::path output_dir = problem.output_directory();
              std::ostringstream          cmd;
              cmd << "bash \"" << plot_script.string() << "\" \""
                  << output_dir.string() << "\"";
              const int ret = std::system(cmd.str().c_str());
              if (ret != 0)
                std::cerr << "Warning: post-run plot script failed." << std::endl;
            }
          else
            {
              std::cerr << "Warning: plot script not found: "
                        << plot_script.string() << std::endl;
            }
        }
    }
  catch (const std::exception &e)
    {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
    }

  return 0;
}
