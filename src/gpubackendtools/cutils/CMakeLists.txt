# =============================
# ==== BACKEND COMPILATION ====
# =============================

include(cmake_functions.cmake)
set(HERE_CUDA_ARCH ${GBT_CUDA_ARCH})

# * * * * * * * * * * * * * * * * * * * * * * * * *
# * * Helper functions to define backend option * *
# * * * * * * * * * * * * * * * * * * * * * * * * *

# In the project root CMakeLists.txt, we defined a "gbt" interface
# target with properties WITH_CPU and WITH_GPU defining whether the CPU and a
# GPU backend need to be compiled. Let's retrieve these information here:
get_target_property(GBT_WITH_CPU gpubackendtools WITH_CPU)
get_target_property(GBT_WITH_GPU gpubackendtools WITH_GPU)

# Adapter to let inplace editable install work: the compiled backend will be
# placed into the source-tree in the 'src' directory:
if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${gpubackendtools_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()


# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *


# ----------------
# --- interp ---
# ----------------

# I. Declare the CPU backend
if(GBT_WITH_CPU)
  add_custom_command(
    OUTPUT "Interpolate.cxx"
    COMMENT "Copy Interpolate.cu to Interpolate.cxx"
    COMMAND
      ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Interpolate.cu"
      "${CMAKE_CURRENT_BINARY_DIR}/Interpolate.cxx"
    DEPENDS "Interpolate.cu"
    VERBATIM)

  # interp needs to link against LAPACKE. USe the following method to detect
  # it and obtain ${GBT_LAPACKE_LIBS}. In case of failure, it will stop the
  # processing execution.
  enable_language(Fortran)
  get_lapacke()

  add_library(gpubackendtools_cpu_interp_static STATIC Interpolate.cxx)
  apply_cpu_backend_common_options(interp gbt gpubackendtools true)
  # target_include_directories(gpubackendtools_cpu_interp_static PUBLIC ${GBT_CUTILS})
  target_sources(gpubackendtools_cpu_interp_static PUBLIC FILE_SET HEADERS FILES
                                        Interpolate.hh)
  target_link_libraries(gpubackendtools_cpu_interp_static LINK_PUBLIC ${GBT_LAPACKE_LIBS}
                        ${GBT_LAPACKE_EXTRA_LIBS})
  if(GBT_LAPACKE_LINK_DIRS)
    target_link_directories(gpubackendtools_cpu_interp_static PUBLIC
                            ${GBT_LAPACKE_LINK_DIRS})
  endif()

  nanobind_add_module(gpubackendtools_cpu_interp gbt_binding.cxx)
  apply_cpu_backend_common_options(interp gbt gpubackendtools false)
  target_link_libraries(gpubackendtools_cpu_interp PRIVATE gpubackendtools_cpu_interp_static)
  # target_include_directories(gpubackendtools_cpu_interp PUBLIC ${GBT_CUTILS})
  target_sources(gpubackendtools_cpu_interp PUBLIC FILE_SET HEADERS FILES
                                      gbt_binding.hpp)
  target_link_libraries(gpubackendtools_cpu_interp LINK_PUBLIC ${GBT_LAPACKE_LIBS}
                        ${GBT_LAPACKE_EXTRA_LIBS})
  if(GBT_LAPACKE_LINK_DIRS)
    target_link_directories(gpubackendtools_cpu_interp PUBLIC
                            ${GBT_LAPACKE_LINK_DIRS})
  endif()

endif()

# III. Declare the GPU backend
if(GBT_WITH_GPU)
  add_library(gpubackendtools_gpu_interp_gpu_static STATIC Interpolate.cu)
  apply_gpu_backend_common_options(interp_gpu gbt gpubackendtools true)
  # target_include_directories(gpubackendtools_gpu_interp_gpu_static PUBLIC ${GBT_CUTILS})
  target_sources(gpubackendtools_gpu_interp_gpu_static PUBLIC FILE_SET HEADERS FILES
                                        Interpolate.hh)
  
  nanobind_add_module(gpubackendtools_gpu_interp gbt_binding.cxx)
  apply_gpu_backend_common_options(interp gbt gpubackendtools false)
  target_link_libraries(gpubackendtools_gpu_interp PRIVATE gpubackendtools_gpu_interp_gpu_static)
  # target_include_directories(gpubackendtools_gpu_interp PUBLIC ${GBT_CUTILS})
  target_sources(gpubackendtools_gpu_interp PUBLIC FILE_SET HEADERS FILES
                                      gbt_binding.hpp)

endif()
