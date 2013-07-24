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

LIST( APPEND OW_LIB_OPENWALNUT ${LABP_CORE} )


# -----------------------------------------------------------------------------------------------------------------------------------------------
# Switch for float/double computation
# -----------------------------------------------------------------------------------------------------------------------------------------------

OPTION( LABP_FLOAT_COMPUTATION "Use float instead of double." )
IF( LABP_FLOAT_COMPUTATION )
	ADD_DEFINITIONS( -DLABP_FLOAT_COMPUTATION )
ENDIF( LABP_FLOAT_COMPUTATION )


# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# Some modules need third party libraries. We provide them if they are available
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

include( "modules-dependencies.cmake" )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# Toolbox Setup
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Toolbox mechanism. This is a nice possibility to turn on/off several modules. This can speed up build times if unneeded modules are turned off.
option( LABP_MODULE_TOOLBOX_ALL             "Turn all toolboxes on." OFF )
option( LABP_MODULE_TOOLBOX_BASE            "All the basic functionality modules. These modules are useful and needed for common visualizatuion tasks." ON )

# Include needed lists
if( LABP_MODULE_TOOLBOX_BASE OR LABP_MODULE_TOOLBOX_ALL )
  include( modules-base.toolbox )
endif( LABP_MODULE_TOOLBOX_BASE OR LABP_MODULE_TOOLBOX_ALL )


# optional user toolbox which is not unter version control allowing all to only compile their modules needed.
include( modules-user.toolbox OPTIONAL )

