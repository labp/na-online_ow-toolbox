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

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Some common setup
# -----------------------------------------------------------------------------------------------------------------------------------------------

SET( LibName ${LABP_CORE} )
SET( libfiffio_dir ${PROJECT_SOURCE_DIR}/../resources/platformDependent/libfiffio )

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Switch for float/double computation
# -----------------------------------------------------------------------------------------------------------------------------------------------

OPTION( LABP_FLOAT_COMPUTATION "Use float instead of double." )
IF( LABP_FLOAT_COMPUTATION )
	ADD_DEFINITIONS( -DLABP_FLOAT_COMPUTATION )
ENDIF( LABP_FLOAT_COMPUTATION )


# -----------------------------------------------------------------------------------------------------------------------------------------------
# Search own dependencies
# -----------------------------------------------------------------------------------------------------------------------------------------------

SET( ADDITIONAL_LIBS "" )


# -----------------------------------------------------------------------------------------------------------------------------------------------
# Add CUDA header
# -----------------------------------------------------------------------------------------------------------------------------------------------

FIND_PACKAGE( CUDA QUIET )
IF( CUDA_FOUND AND OW_USE_CUDA )
	ADD_DEFINITIONS( -DCUDA_FOUND )
	INCLUDE_DIRECTORIES( ${CUDA_TOOLKIT_INCLUDE} )
	INCLUDE_DIRECTORIES( "${CUDA_SDK_ROOT_DIR}/common/inc" )
ENDIF( CUDA_FOUND AND OW_USE_CUDA )


# -----------------------------------------------------------------------------------------------------------------------------------------------
# Add MNE Library for FIFF reader
# -----------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Order of static libs is important. Wrong order can produce undefined references and symbols!
#       We need the MNE dependencies twice :(

FIND_PACKAGE( MNE REQUIRED )
INCLUDE_DIRECTORIES( ${MNE_INCLUDE_DIR} )
LINK_DIRECTORIES( ${MNE_LIBRARY_DIR} )
LIST( APPEND ADDITIONAL_LIBS ${MNE_FIFF_LIBRARY} ${MNE_LIBRARY} ${MNE_UTILS_LIBRARY} ${MNE_FS_LIBRARY} )


# -----------------------------------------------------------------------------------------------------------------------------------------------
# Add static Qt5 for MNE Library
# -----------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Need of static linking due to conflicts with Qt4 of OpenWalnut (runtime error).
# NOTE: Order of Qt5 static libs is important. Wrong order can produce undefined references and symbols!

FIND_PACKAGE( Qt5Static REQUIRED )
INCLUDE_DIRECTORIES( ${QT5_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${QT5_QTCORE_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${QT5_QTNETWORK_INCLUDE_DIR} )
LINK_DIRECTORIES( ${QT5_STATIC_LIBRARY_DIR} )
LIST( APPEND ADDITIONAL_LIBS ${QT5_STATIC_QTCONCURRENT_LIBRARY} ${QT5_STATIC_QTNETWORK_LIBRARY} ${QT5_STATIC_QTCORE_LIBRARY} )
# Libraries needed due to Qt5 static linking
LIST( APPEND ADDITIONAL_LIBS pthread rt z icudata icui18n icuuc glib-2.0 dl )


# -----------------------------------------------------------------------------------------------------------------------------------------------
# Add MNE Library for FIFF reader
# -----------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Order of static libs is important. Wrong order can produce undefined references and symbols!

FIND_PACKAGE( MNE REQUIRED )
INCLUDE_DIRECTORIES( ${MNE_INCLUDE_DIR} )
LINK_DIRECTORIES( ${MNE_LIBRARY_DIR} )
LIST( APPEND ADDITIONAL_LIBS ${MNE_FIFF_LIBRARY} ${MNE_LIBRARY} ${MNE_UTILS_LIBRARY} ${MNE_FS_LIBRARY} )


# -----------------------------------------------------------------------------------------------------------------------------------------------
# OpenWalnut dependencies
# -----------------------------------------------------------------------------------------------------------------------------------------------

INCLUDE_DIRECTORIES( ${libfiffio_dir}/include )

LIST( APPEND ADDITIONAL_LIBS ${OW_LIB_OPENWALNUT} )
LIST( APPEND ADDITIONAL_LIBS ${libfiffio_dir}/linux_x86_64/libfiffio.so )
LIST( APPEND ADDITIONAL_LIBS ${Boost_LIBRARIES} ${OPENSCENEGRAPH_LIBRARIES} ) # ${CMAKE_STANDARD_LIBRARIES} ${CMAKE_DL_LIBS} ${OPENGL_gl_LIBRARY} 


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Setup for compilation
# ---------------------------------------------------------------------------------------------------------------------------------------------------

COLLECT_COMPILE_FILES( "${CMAKE_CURRENT_SOURCE_DIR}" TARGET_CPP_FILES TARGET_H_FILES TARGET_TEST_FILES )

ADD_LIBRARY( ${LibName} SHARED ${TARGET_CPP_FILES} ${TARGET_H_FILES} )
TARGET_LINK_LIBRARIES( ${LibName} ${ADDITIONAL_LIBS} )

# Do not forget the install targets
SETUP_LIB_INSTALL( ${LibName} ${OW_LIBRARY_DIR_RELATIVE}/.. ${LABP_CORE} )


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Test Setup
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Setup tests of this target
SETUP_TESTS( "${TARGET_TEST_FILES}" "${LibName}" "" )

