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
SET( libfiffio_dir ${PROJECT_SOURCE_DIR}/../resources/platformDependent/libfiffio )


# --------------------------------------------------------------------------------------------------------------------------------
# Search own dependencies
# --------------------------------------------------------------------------------------------------------------------------------

SET( ADDITIONAL_LIBS "" )

INCLUDE_DIRECTORIES( ${libfiffio_dir}/include )
LIST( APPEND ADDITIONAL_LIBS ${libfiffio_dir}/linux_x86_64/libfiffio.so )

# --------------------------------------------------------------------------------------------------------------------------------
# Add MNE Library headers
# --------------------------------------------------------------------------------------------------------------------------------

FIND_PACKAGE( MNE REQUIRED )
INCLUDE_DIRECTORIES( ${MNE_INCLUDE_DIR} )

FIND_PACKAGE( Qt5Core REQUIRED )
INCLUDE_DIRECTORIES( ${Qt5_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES( ${Qt5Core_INCLUDE_DIRS} )


# --------------------------------------------------------------------------------------------------------------------------------
# Setup for compilation
# --------------------------------------------------------------------------------------------------------------------------------

# Let this function do the job. It sets up tests and copies shaders automatically.
# It additionally configures the stylecheck mechanism for this module.
SETUP_MODULE( ${MODULE_NAME}                # name of the module
             "${CMAKE_CURRENT_SOURCE_DIR}"  # where to find the source ${CMAKE_CURRENT_SOURCE_DIR} is a good idea here mostly
             "${ADDITIONAL_LIBS}"           # does your module need additional libs to compile?
             ""                             # do you want to exclude files from stylechecking? (externals for example)
)
