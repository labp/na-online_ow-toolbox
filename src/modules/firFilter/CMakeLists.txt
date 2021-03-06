#---------------------------------------------------------------------------
#
# Project: NA-Online ( http://www.labp.htwk-leipzig.de )
#
# Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
#
# This file is part of NA-Online.
#
# NA-Online is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NA-Online is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
#
#---------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------
# Some common setup
# --------------------------------------------------------------------------------------------------------------------------------

# we use the directory name as module name
GET_FILENAME_COMPONENT( MODULE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME )

# setup the target directories and names
SET( MODULE_TARGET_DIR_RELATIVE ${OW_MODULE_DIR_RELATIVE}/${MODULE_NAME} )
SET( MODULE_TARGET_DIR ${PROJECT_BINARY_DIR}/${MODULE_TARGET_DIR_RELATIVE} )
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MODULE_TARGET_DIR} )
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MODULE_TARGET_DIR} )
SET( MODULE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )


# --------------------------------------------------------------------------------------------------------------------------------
# Add sources as target
# --------------------------------------------------------------------------------------------------------------------------------

# Collect the compile-files for this target
COLLECT_COMPILE_FILES( "${MODULE_SOURCE_DIR}" TARGET_CPP_FILES TARGET_H_FILES TARGET_TEST_FILES )


# --------------------------------------------------------------------------------------------------------------------------------
# Add CUDA library and code
# --------------------------------------------------------------------------------------------------------------------------------

IF( CUDA_FOUND AND OW_USE_CUDA )
	ADD_DEFINITIONS( -DFOUND_CUDA )
    FILE( GLOB_RECURSE CUDA_SRC "*.cu" )
    CUDA_ADD_LIBRARY( ${MODULE_NAME} SHARED ${TARGET_CPP_FILES} ${TARGET_H_FILES} ${CUDA_SRC} )
    TARGET_LINK_LIBRARIES( ${MODULE_NAME} ${OW_LIB_OPENWALNUT} ${CUDA_LIBRARIES} ${Boost_LIBRARIES} )
ELSE()
    # Remove files that do including cuda stuff
	LIST( REMOVE_ITEM TARGET_H_FILES ${CMAKE_CURRENT_SOURCE_DIR}/WFIRFilterCuda.h )
	LIST( REMOVE_ITEM TARGET_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/WFIRFilterCuda.cpp )
    # LIST( REMOVE_ITEM TARGET_H_FILES ${CMAKE_CURRENT_SOURCE_DIR}/WFIRFilterCuda.cu )

    # Build the module without the cuda codes
    ADD_LIBRARY( ${MODULE_NAME} SHARED ${TARGET_CPP_FILES} ${TARGET_H_FILES} )
    TARGET_LINK_LIBRARIES( ${MODULE_NAME} ${OW_LIB_OPENWALNUT} ${Boost_LIBRARIES} )
ENDIF()


# Set the version of the library.
SET_TARGET_PROPERTIES( ${MODULE_NAME} PROPERTIES
    VERSION ${OW_LIB_VERSION}
    SOVERSION ${OW_SOVERSION} )

# Do not forget the install targets
SETUP_LIB_INSTALL( ${MODULE_NAME} ${MODULE_TARGET_DIR_RELATIVE} "MODULES" )


# --------------------------------------------------------------------------------------------------------------------------------
# Test Setup
# --------------------------------------------------------------------------------------------------------------------------------

# Setup tests of this target
IF( NOT CUDA_FOUND OR NOT OW_USE_CUDA )
	LIST( REMOVE_ITEM TARGET_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test/WFIRFilterCuda_test.h )
ENDIF()
SETUP_TESTS( "${TARGET_TEST_FILES}" "${MODULE_NAME}" "${_MODULE_DEPENDENCIES}" )

# --------------------------------------------------------------------------------------------------------------------------------
# Style Checker
# --------------------------------------------------------------------------------------------------------------------------------

# setup the style checker. Ignore the platform specific stuff.
SETUP_STYLECHECKER( "${MODULE_NAME}"
                    "${TARGET_CPP_FILES};${TARGET_H_FILES};${TARGET_TEST_FILES};${TARGET_GLSL_FILES}"  # add all these files to the style checker
                    "${_MODULE_STYLE_EXCLUDES}" )                                                      # exclude some ugly files

