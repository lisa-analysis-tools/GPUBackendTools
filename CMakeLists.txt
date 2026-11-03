# ====================================
# == GBT project base configuration ==
# ====================================

# ---- CMake related definitions ----
cmake_minimum_required(VERSION 3.23...3.31)

# ---- Main project definition ----
project(gpubackendtools VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX Fortran) #@ADD_CUDA_HERE

# ---- Find required dependencies ----
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
# Phase 3M (2026-06-04): sprint-wide pybind11 -> nanobind migration.
# nanobind ships its CMake config under `${nanobind_DIR}` resolved by
# `python -m nanobind --cmake_dir`. Resolve here once so every target
# can call `nanobind_add_module(...)` without re-shelling.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

# This line defines the install directory variables
include(GNUInstallDirs)

# ---- Import project-specific CMake functions ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- Define a list of GBT specific options ----
set_property(GLOBAL PROPERTY GBT_OPTION_NAMES)

# ---- Define project specific options ----

# GBT_WITH_GPU controls the activation of both GPU and CPU backend compilations.
# Its allowed values are:
#
# * AUTO: enable CPU backend, and enable GPU backend if CUDA toolchain is found
#   in environment, otherwise disable it
# * ON: enable CPU backend, and enable GPU backend, fail if CUDA toolchain is
#   not available
# * OFF: enable CPU backend and disable GPU backend
# * ONLY: disable CPU backend and enable GPU backend (used in plugin wheel build
#   process)
# * BARE: disable both CPU and GPU backends
set(GBT_WITH_GPU "AUTO" CACHE STRING "Whether to compile GPU backend")
set_property(CACHE GBT_WITH_GPU PROPERTY STRINGS "AUTO" "ON" "OFF" "ONLY"
                                         "BARE")
set_property(GLOBAL APPEND PROPERTY GBT_OPTION_NAMES GBT_WITH_GPU)

# GBT_CUDA_ARCH will be passed as the CUDA_ARCHITECTURES property for the CUDA
# backend if it is compiled. See the documentation of CUDA_ARCHITECTURES:
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(GBT_CUDA_ARCH "native"
    CACHE STRING "CUDA Architecture targetted for GBT compilation (see doc of \
          CMAKE_CUDA_ARCHITECTURES).")
set_property(GLOBAL APPEND PROPERTY GBT_OPTION_NAMES GBT_CUDA_ARCH)

set(GBT_MARCH "native"
    CACHE STRING "Value of the -march compiler option if supported by compiler")
set_property(GLOBAL APPEND PROPERTY GBT_OPTION_NAMES GBT_MARCH)


# GBT_LAPACKE_DETECT_WITH sets the tool used to try to detect LAPACKE locally.
# Possible values are:
#
# * AUTO: attempts PKGCONFIG and, in case of failure, CMAKE
# * CMAKE: use CMake find_package() mechanism (try to find
#   'lapacke-config.cmake' in CMAKE_PREFIX_PATH or in standard system locations)
# * PKGCONFIG: use pkgconfig (which must be installed), try to find 'lapacke.pc'
#   and its dependencies in standard system locations or in PKG_CONFIG_PATH
# * DISABLE (or any other value): disable detection of local lapacke
#   installation
set(GBT_LAPACKE_DETECT_WITH "AUTO" CACHE STRING "Tool used to locate LAPACKE.")
set_property(CACHE GBT_LAPACKE_DETECT_WITH PROPERTY STRINGS "AUTO" "CMAKE"
                                                    "PKGCONFIG" "DISABLE")
set_property(GLOBAL APPEND PROPERTY GBT_OPTION_NAMES GBT_LAPACKE_DETECT_WITH)

# GBT_LAPACKE_FETCH controls whether LAPACKE sources will be downloaded and
# compiled with fast LISA response backends. Possible values are:
#
# * AUTO: will try to locate LAPACKE locally and, in case of failure, will
#   enable the fetch mechanism
# * ON: will always fetch LAPACKE sources and compile them, even if already
#   present locally
# * OFF: will disable LAPACKE fetching mechanism
set(GBT_LAPACKE_FETCH "AUTO" CACHE STRING
                                   "Whether to download and compile LAPACK(E)")
set_property(CACHE GBT_LAPACKE_FETCH PROPERTY STRINGS "AUTO" "ON" "OFF")
set_property(GLOBAL APPEND PROPERTY GBT_OPTION_NAMES GBT_LAPACKE_FETCH)

# Explicitly set the path to gfortran
# set(CMAKE_Fortran_COMPILER "/opt/homebrew/bin/gfortran") 

# GBT_LAPACKE_EXTRA_LIBS sets extra libraries that must be linked when linking
# LAPACKE. For example, if importing the CPU backend fails and complains about
# missing "gfortran" symbols, set GBT_LAPACKE_EXTRA_LIBS=gfortran. Note that if
# this option is unset, it will be automatically set to "gfortran" if that
# library is linkable.
#
# GBT_LAPACKE_LINK_DIRS is a companion option: extra directories added to
# the linker search path when resolving GBT_LAPACKE_EXTRA_LIBS. Auto-detected
# from CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES so builds on Apple Silicon
# (Homebrew gcc's /opt/homebrew/Cellar/gcc/<ver>/lib/gcc/current), or any
# other system where libgfortran is not on clang's default library search
# path, resolve the Fortran runtime symbols pulled in by liblapack.a
# (__gfortran_st_write, __gfortran_concat_string, ...). Explicit user
# values are always honored.
include(CheckLinkerFlag)

set(DEFAULT_EXTRA_LIBS "")
set(DEFAULT_EXTRA_LINK_DIRS "")

# Preferred path: the Fortran language is already enabled by the top-level
# project(... LANGUAGES CXX Fortran) call, so CMake has probed the Fortran
# compiler and populated CMAKE_Fortran_IMPLICIT_LINK_{DIRECTORIES,LIBRARIES}
# with exactly the directories/libs the Fortran driver would use. That is
# the authoritative answer; -lgfortran alone under clang++ does not otherwise
# see Homebrew gcc's runtime dir on Apple Silicon.
if(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES)
  list(FIND CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "gfortran" _gbt_gf_idx)
  if(NOT _gbt_gf_idx EQUAL -1)
    set(DEFAULT_EXTRA_LIBS "gfortran")
    set(DEFAULT_EXTRA_LINK_DIRS "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
    message(STATUS "gfortran available via Fortran compiler implicit link "
                   "directories: ${DEFAULT_EXTRA_LINK_DIRS}")
  endif()
  unset(_gbt_gf_idx)
endif()

# Fallback probe: bare -lgfortran with clang++'s default search path.
# Covers Linux distros where libgfortran lives under /usr/lib/gcc/... and
# systems without the Fortran language enabled.
if(NOT DEFAULT_EXTRA_LIBS)
  check_linker_flag(CXX "-lgfortran" GFORTRAN_AVAILABLE)
  if(GFORTRAN_AVAILABLE)
    message(STATUS "gfortran available on default linker search path.")
    set(DEFAULT_EXTRA_LIBS "gfortran")
  else()
    message(STATUS "gfortran NOT available.")
  endif()
  unset(GFORTRAN_AVAILABLE)
endif()

set(GBT_LAPACKE_EXTRA_LIBS "${DEFAULT_EXTRA_LIBS}"
    CACHE STRING "Extra libs to link to when\
linking against LAPACKE.")
set(GBT_LAPACKE_LINK_DIRS "${DEFAULT_EXTRA_LINK_DIRS}"
    CACHE STRING "Extra directories added to the linker search path when\
 resolving GBT_LAPACKE_EXTRA_LIBS.")
set_property(GLOBAL APPEND PROPERTY GBT_OPTION_NAMES GBT_LAPACKE_EXTRA_LIBS)
set_property(GLOBAL APPEND PROPERTY GBT_OPTION_NAMES GBT_LAPACKE_LINK_DIRS)
unset(DEFAULT_EXTRA_LIBS)
unset(DEFAULT_EXTRA_LINK_DIRS)


# ---- Phony target for project specific properties ----
add_library(gpubackendtools INTERFACE)

# ---- Enable building the CPU version of backends by default ----
set_target_properties(gpubackendtools PROPERTIES WITH_CPU ON)

# ---- Test whether the GBT_MARCH option is supported by CXX compiler ----
include(CheckCXXCompilerFlag)
set(GBT_MARCH_CXX_OPT "-march=${GBT_MARCH}")
check_cxx_compiler_flag("${GBT_MARCH_CXX_OPT}" CXX_COMPILER_SUPPORTS_GBT_MARCH)
if(CXX_COMPILER_SUPPORTS_GBT_MARCH)
  set_property(TARGET gpubackendtools PROPERTY CXX_MARCH
                                                 "${GBT_MARCH_CXX_OPT}")
  message(STATUS "The CXX compiler supports option '${GBT_MARCH_CXX_OPT}'.")
else()
  message(
    WARNING "The CXX compiler does not support option '${GBT_MARCH_CXX_OPT}'. \
      It will be ignored.")
endif()

# ---- Optionnally check if GPU is supported ----
if(GBT_WITH_GPU STREQUAL "AUTO")
  if(DEFINED ENV{READTHEDOCS})
    message(
      STATUS
        "GBT GPU backend generation is disabled in ReadTheDocs build context.")
    set(GBT_WITH_GPU "OFF")
  endif()
endif()
if(GBT_WITH_GPU STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
  endif()
  if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    message(
      STATUS
        "GBT GPU support was set to AUTO and will be turned ON as CUDA and \
        CUDA Toolkit are available with CUDA version \
        ${CUDAToolkit_VERSION_MAJOR}.")
    set_target_properties(gpubackendtools PROPERTIES WITH_GPU ON)
  else()
    message(
      STATUS
        "GBT GPU support was set to AUTO and will be turned OFF as CUDA and \
        CUDA Toolkit are not found (CMAKE_CUDA_COMPILER:${CMAKE_CUDA_COMPILER} \
        CUDAToolkit_FOUND:${CUDAToolkit_FOUND}).")
    set_target_properties(gpubackendtools PROPERTIES WITH_GPU OFF)
  endif()
elseif(GBT_WITH_GPU STREQUAL "ONLY")
  message(
    STATUS
      "GBT GPU support is set to ON and CPU support is set to OFF (use only to \
      build a standalone plugin).")
  set_target_properties(gpubackendtools PROPERTIES WITH_GPU ON)
  set_target_properties(gpubackendtools PROPERTIES WITH_CPU OFF)
elseif(GBT_WITH_GPU STREQUAL "BARE")
  message(STATUS "GBT GPU and CPU support are disabled (use only to build a \
      non-functional pure-python release).")
  set_target_properties(gpubackendtools PROPERTIES WITH_GPU OFF)
  set_target_properties(gpubackendtools PROPERTIES WITH_CPU OFF)
else()
  message(STATUS "GBT GPU support is set to ${GBT_WITH_GPU}.")
  set_target_properties(gpubackendtools PROPERTIES WITH_GPU ${GBT_WITH_GPU})
endif()

# ---- Handle GPU support ----
get_target_property(_GBT_WITH_GPU gpubackendtools WITH_GPU)
if(_GBT_WITH_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# ---- Include sources ----
add_subdirectory(src)

# ---- Let CMake handle copying CITATION.cff into wheel ----
if(SKBUILD_STATE STREQUAL "sdist" OR SKBUILD_STATE STREQUAL "wheel")
  if(_GBT_WITH_CPU)
    # A bare install does not need to embed CITATION.cff
    install(FILES CITATION.cff DESTINATION gpubackendtools)
  endif()
endif()

# ---- Print the list of options used for this build ----
get_property(ALL_GBT_OPTION_NAMES GLOBAL PROPERTY GBT_OPTION_NAMES)

# We assume here that:
#
# 1. The list of options is not empty (at least one GBT option was registered)
# 2. All registered options exist as cache variables (since they are all defined
#    in this same file before being registered)
# 3. All cache variables have values (CMake cache variables always have a value,
#    even if it's an empty string)
message(STATUS "GBT options used for this build:")
foreach(gpubackendtools_option_name ${ALL_GBT_OPTION_NAMES})
  get_property(GBT_OPTION_VALUE CACHE ${gpubackendtools_option_name} PROPERTY VALUE)
  message(STATUS "  ${gpubackendtools_option_name}: ${GBT_OPTION_VALUE}")
endforeach()
