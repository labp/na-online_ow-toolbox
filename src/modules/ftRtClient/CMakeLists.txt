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
# Add FieldTrip Library for Real Time Streaming
# -----------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Order of static libs is important. Wrong order can produce undefined references and symbols!

FIND_PACKAGE( FtBuffer REQUIRED )
INCLUDE_DIRECTORIES( ${FT_BUFFER_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${FT_CLIENT_INCLUDE_DIR} )
LINK_DIRECTORIES( ${FT_BUFFER_LIBRARY_DIR} )
LIST( APPEND ADDITIONAL_LIBS ${FT_BUFFER_LIBRARY} )



# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Setup for compilation
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Let this function do the job. It sets up tests and copies shaders automatically. It additionally configures the stylecheck mechanism for this
# module.
SETUP_MODULE( ${MODULE_NAME}                # name of the module
             "${CMAKE_CURRENT_SOURCE_DIR}"  # where to find the source ${CMAKE_CURRENT_SOURCE_DIR} is a good idea here mostly
             "${ADDITIONAL_LIBS}"           # does your module need additional libs to compile?
             ""                             # do you want to exclude files from stylechecking? (externals for example)
)
