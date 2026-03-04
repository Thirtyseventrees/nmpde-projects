cmake_minimum_required(VERSION 3.12.0)
project(project-wave2d LANGUAGES CXX C)

# Prefer the same build environment as labs-history:
#   module load gcc-glibc dealii
#
# Some setups set CMAKE_FIND_PACKAGE_PREFER_CONFIG=ON, making CMake look for
# BoostConfig.cmake. Force the classic FindBoost.cmake module instead.
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG OFF CACHE BOOL "" FORCE)
set(Boost_NO_BOOST_CMAKE ON CACHE BOOL "" FORCE)

include(common/cmake-common.cmake)

add_library(Wave2D STATIC src/Wave2D.cpp)
target_compile_definitions(Wave2D PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
deal_ii_setup_target(Wave2D)

add_executable(main src/main.cpp)
target_link_libraries(main Wave2D)
target_compile_definitions(main PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
deal_ii_setup_target(main)

add_custom_target(clean-results
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_SOURCE_DIR}/result"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/result"
  COMMENT "Remove all files under result/"
)
