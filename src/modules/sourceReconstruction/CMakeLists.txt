#---------------------------------------------------------------------------
#
# Project: OpenWalnut ( http://www.openwalnut.org )
#
# Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS
# For more information see http:#www.openwalnut.org/copying
#
# This file is part of OpenWalnut.
#
# OpenWalnut is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenWalnut is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with OpenWalnut. If not, see <http:#www.gnu.org/licenses/>.
#
#---------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Some common setup
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# we use the directory name as module name
GET_FILENAME_COMPONENT( MODULE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Search own dependencies
# ---------------------------------------------------------------------------------------------------------------------------------------------------

SET( ADDITIONAL_LIBS "" )

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------------------------------------------------------------------------------

IF( CMAKE_BUILD_TYPE MATCHES Release AND CMAKE_COMPILER_IS_GNUCXX )
	ADD_DEFINITIONS( -O2 )
ENDIF()
IF( CMAKE_BUILD_TYPE MATCHES Debug AND CMAKE_COMPILER_IS_GNUCXX )
	ADD_DEFINITIONS( -O0 )
ENDIF()

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Add SuperLU Library for LU decomposition with sparse matrices
# -----------------------------------------------------------------------------------------------------------------------------------------------

FIND_PACKAGE( SuperLU REQUIRED )

ADD_DEFINITIONS( ${SUPERLU_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${SUPERLU_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${SUPERLU_LIBRARY_DIRS} )
LIST( APPEND ADDITIONAL_LIBS ${SUPERLU_LIBRARIES} )

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Add CUDA libraries
# -----------------------------------------------------------------------------------------------------------------------------------------------

IF( CUDA_FOUND AND OW_USE_CUDA )
	ADD_DEFINITIONS( -DFOUND_CUDA )
	INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS} )
	LINK_DIRECTORIES( ${CUDA_LIBRARY_DIRS} )
    LIST( APPEND ADDITIONAL_LIBS ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} )
ENDIF()
# Remove CUDA files in SETUP_MODULE

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Setup for compilation
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Copy of SETUP_MODULE due to delete CUDA files #
# Set function arguments instead of calling function
SET( _MODULE_NAME ${MODULE_NAME} )
SET( _MODULE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
SET( _MODULE_DEPENDENCIES ${ADDITIONAL_LIBS} )
SET( _MODULE_STYLE_EXCLUDES "")

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Some common setup
# -----------------------------------------------------------------------------------------------------------------------------------------------

# setup the target directories and names
SET( MODULE_NAME ${_MODULE_NAME} )
SET( MODULE_TARGET_DIR_RELATIVE ${OW_MODULE_DIR_RELATIVE}/${MODULE_NAME} )
SET( MODULE_TARGET_RESOURCE_DIR_RELATIVE ${OW_SHARE_DIR_RELATIVE}/modules/${MODULE_NAME} )
SET( MODULE_TARGET_DIR ${PROJECT_BINARY_DIR}/${MODULE_TARGET_DIR_RELATIVE} )
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MODULE_TARGET_DIR} )
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MODULE_TARGET_DIR} )
SET( MODULE_SOURCE_DIR ${_MODULE_SOURCE_DIR} )

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Add sources as target
# -----------------------------------------------------------------------------------------------------------------------------------------------

# Collect the compile-files for this target
COLLECT_COMPILE_FILES( "${MODULE_SOURCE_DIR}" TARGET_CPP_FILES TARGET_H_FILES TARGET_TEST_FILES )
# Remove CUDA files
IF( NOT CUDA_FOUND OR NOT OW_USE_CUDA )
    # Remove files which contain CUDA stuff
	LIST( REMOVE_ITEM TARGET_H_FILES ${CMAKE_CURRENT_SOURCE_DIR}/WSourceReconstructionCuda.h )
	LIST( REMOVE_ITEM TARGET_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/WSourceReconstructionCuda.cpp )
ENDIF()

# Setup the target
ADD_LIBRARY( ${MODULE_NAME} SHARED ${TARGET_CPP_FILES} ${TARGET_H_FILES} )
TARGET_LINK_LIBRARIES( ${MODULE_NAME} ${CMAKE_STANDARD_LIBRARIES} ${OW_LIB_OPENWALNUT} ${Boost_LIBRARIES} ${OPENGL_gl_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES} ${_MODULE_DEPENDENCIES} )

# Set the version of the library.
SET_TARGET_PROPERTIES( ${MODULE_NAME} PROPERTIES
    VERSION ${OW_LIB_VERSION}
    SOVERSION ${OW_SOVERSION}
)

# Do not forget the install targets
SETUP_LIB_INSTALL( ${MODULE_NAME} ${MODULE_TARGET_DIR_RELATIVE} "MODULES" )

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Test Setup
# -----------------------------------------------------------------------------------------------------------------------------------------------

# Setup tests of this target
SETUP_TESTS( "${TARGET_TEST_FILES}" "${MODULE_NAME}" "${_MODULE_DEPENDENCIES}" )

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Copy Shaders
# -----------------------------------------------------------------------------------------------------------------------------------------------

COLLECT_SHADER_FILES( ${MODULE_SOURCE_DIR} TARGET_GLSL_FILES )
SETUP_SHADERS( "${TARGET_GLSL_FILES}" "${MODULE_TARGET_RESOURCE_DIR_RELATIVE}/shaders" "MODULES" )

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Copy Additional Resources
# -----------------------------------------------------------------------------------------------------------------------------------------------
SETUP_RESOURCES_GENERIC( "${MODULE_SOURCE_DIR}/resources" ${MODULE_TARGET_RESOURCE_DIR_RELATIVE} "${_MODULE_NAME}" "MODULES" )

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Style Checker
# -----------------------------------------------------------------------------------------------------------------------------------------------

# setup the stylechecker. Ignore the platform specific stuff.
SETUP_STYLECHECKER( "${MODULE_NAME}"
                    "${TARGET_CPP_FILES};${TARGET_H_FILES};${TARGET_TEST_FILES};${TARGET_GLSL_FILES}"  # add all these files to the stylechecker
                    "${_MODULE_STYLE_EXCLUDES}" )                                                      # exlude some ugly files
