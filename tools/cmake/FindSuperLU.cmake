# ADD (pieloth): Taken from https://svn.dune-project.org/svn/dune-istl/branches/cmake/cmake/modules/FindSuperLU.cmake
#
# Module that checks whether SuperLU is available and usable.
# SuperLU must be a version released after the year 2005.
#
# Sets the follwing variable:
#
# SUPERLU_FOUND           True if SuperLU available and usable.
# SUPERLU_MIN_VERSION_4_3 True if SuperLU version >= 4.3.
# SUPERLU_WITH_VERSION    Human readable string containing version information.
# SUPERLU_INCLUDE_DIRS    Path to the SuperLU include dirs.
# SUPERLU_LIBRARIES       Name to the SuperLU library.
#
find_package(BLAS QUIET REQUIRED)
if(NOT BLAS_FOUND AND REQUIRED)
  message("BLAS not found but required for SuperLU")
  return()
endif(NOT BLAS_FOUND AND REQUIRED)

# look for header files
find_path(SUPERLU_INCLUDE_DIR
  NAMES supermatrix.h
  HINTS ${SUPERLU_DIR}
  PATH_SUFFIXES "superlu" "include/superlu" "include" "SRC"
)

# look for library
find_library(SUPERLU_LIBRARY
  NAMES "superlu_4.3" "superlu_4.2" "superlu_4.1" "superlu_4.0" "superlu_3.1" "superlu_3.0" "superlu"
  HINTS ${SUPERLU_DIR}
  PATH_SUFFIXES "lib" "lib64"
)

# check version specific macros
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${SUPERLU_INCLUDE_DIR})
set(CMAKE_REQUIRED_LIBRARIES ${SUPERLU_LIBRARY})

# check whether "mem_usage_t.expansions" was found in "slu_ddefs.h"
CHECK_C_SOURCE_COMPILES("
#include <slu_ddefs.h>
int main(void)
{
  mem_usage_t mem;
  return mem.expansions;
}"
HAVE_MEM_USAGE_T_EXPANSIONS)

# check whether version is at least 4.3
CHECK_C_SOURCE_COMPILES("
#include <slu_ddefs.h>
int main(void)
{
  return SLU_DOUBLE;
}"
SUPERLU_MIN_VERSION_4_3)
set(CMAKE_REQUIRED_INCLUDES "")
set(CMAKE_REQUIRED_LIBRARIES "")

if(SUPERLU_MIN_VERSION_4_3)
  set(SUPERLU_WITH_VERSION "SuperLU >= 4.3" CACHE STRING
    "Human readable string containing SuperLU version information.")
else()
  set(SUPERLU_WITH_VERSION "SuperLU <= 4.2, post 2005" CACHE STRING
    "Human readable string containing SuperLU version information.")
endif(SUPERLU_MIN_VERSION_4_3)

# behave like a CMake module is supposed to behave
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  "SuperLU"
  DEFAULT_MSG
  SUPERLU_INCLUDE_DIR
  SUPERLU_LIBRARY
)

mark_as_advanced(SUPERLU_INCLUDE_DIRS SUPERLU_LIBRARIES SUPERLU_MIN_VERSION_4_3)

# if both headers and library are found, store results
if(SUPERLU_FOUND)
  set(SUPERLU_INCLUDE_DIRS ${SUPERLU_INCLUDE_DIR})
  set(SUPERLU_LIBRARIES    ${SUPERLU_LIBRARY})
  # log result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
    "Determing location of ${SUPERLU_WITH_VERSION} succeded:\n"
    "Include directory: ${SUPERLU_INCLUDE_DIRS}\n"
    "Library directory: ${SUPERLU_LIBRARIES}\n\n")
  set(SUPERLU_DUNE_COMPILE_FLAGS "-I${SUPERLU_INCLUDE_DIRS}" CACHE STRING 
    "Compile flags used by DUNE when compiling SuperLU programs")
  set(SUPERLU_DUNE_LIBRARIES ${SUPERLU_LIBRARIES} ${BLAS_LIBRARIES} CACHE STRING 
    "Libraries used by DUNE when linking SuperLU programs")
else(SUPERLU_FOUND)
  # log errornous result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
    "Determing location of SuperLU failed:\n"
    "Include directory: ${SUPERLU_INCLUDE_DIRS}\n"
    "Library directory: ${SUPERLU_LIBRARIES}\n\n")
endif(SUPERLU_FOUND)

# set HAVE_SUPERLU for config.h
set(HAVE_SUPERLU SUPERLU_FOUND)

# adds SuperLU flags to the targets
function(add_dune_superlu_flags _targets)
  if(SUPERLU_FOUND)
    foreach(_target ${_targets})
      target_link_libraries(${_target} ${SUPERLU_DUNE_LIBRARIES})
      GET_TARGET_PROPERTY(_props ${_target} COMPILE_FLAGS)
      string(REPLACE "_props-NOTFOUND" "" _props "${_props}")
      SET_TARGET_PROPERTIES(${_target} PROPERTIES COMPILE_FLAGS  
        "${_props} ${SUPERLU_DUNE_COMPILE_FLAGS} -DENABLE_SUPERLU=1")
    endforeach(_target ${_targets})
  endif(SUPERLU_FOUND)
endfunction(add_dune_superlu_flags)

