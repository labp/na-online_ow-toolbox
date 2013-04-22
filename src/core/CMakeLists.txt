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

SET( LibName ${LABP_CORE} )
SET( libfiffio_dir ${PROJECT_SOURCE_DIR}/../resources/platformDependent/libfiffio )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Search own dependencies
# ---------------------------------------------------------------------------------------------------------------------------------------------------

INCLUDE_DIRECTORIES( ${libfiffio_dir}/include )

SET( ADDITIONAL_LIBS "" )

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

