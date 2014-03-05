#---------------------------------------------------------------------------
# This script searches the libraries and header files for FieldTrip Buffer on Ubuntu Linux.
#
# WWW: http://fieldtrip.fcdonders.nl
# Author: Christof Pieloth
#
# The following variables will be filled:
#   * FT_BUFFER_FOUND - if FT_BUFFER header and library found
#   * FT_CLIENT_FOUND - if FT_CLIENT header and library found
#   * FT_BUFFER_INCLUDE_DIR - the path of FT_BUFFER header files
#   * FT_CLIENT_INCLUDE_DIR - the path of FT_CLIENT header files
#   * FT_BUFFER_LIBRARY_DIR - the path to FT_BUFFER libraries
#   * FT_BUFFER_LIBRARY - FT_BUFFER library
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# FT_BUFFER_INCLUDE_DIR
#---------------------------------------------------------------------------

FIND_PATH( FT_BUFFER_INCLUDE_DIR buffer.h HINTS 
        $ENV{FT_BUFFER_INCLUDE_DIR} 
        /opt/include
        /opt/include/ft_buffer
)



#---------------------------------------------------------------------------
# FT_CLIENT_INCLUDE_DIR
#---------------------------------------------------------------------------

FIND_PATH( FT_CLIENT_INCLUDE_DIR FtBuffer.h HINTS 
        $ENV{FT_CLIENT_INCLUDE_DIR} 
        /opt/include
        /opt/include/ft_client
)



#---------------------------------------------------------------------------
# FT_BUFFER_LIBRARY and FT_BUFFER_LIBRARY_DIR
#---------------------------------------------------------------------------

FIND_PATH( FT_BUFFER_LIBRARY_DIR libbuffer.a HINTS 
        $ENV{FT_BUFFER_LIBRARY_DIR} 
        /opt/lib/ft_buffer
        /opt/lib
)

FIND_LIBRARY( FT_BUFFER_LIBRARY buffer HINTS
        $ENV{FT_BUFFER_LIBRARY_DIR} 
        /opt/lib/fieldtrip
        /opt/lib
)

# Retrieve library path for other MNE libraries
get_filename_component( FT_BUFFER_LIBRARY_DIR ${FT_BUFFER_LIBRARY} PATH )



#---------------------------------------------------------------------------
# Finalize setup
#---------------------------------------------------------------------------

SET( FT_BUFFER_FOUND FALSE )
IF ( FT_BUFFER_INCLUDE_DIR AND FT_BUFFER_LIBRARY_DIR )
    SET( FT_BUFFER_FOUND TRUE )
ENDIF ( FT_BUFFER_INCLUDE_DIR AND FT_BUFFER_LIBRARY_DIR )

SET( FT_CLIENT_FOUND FALSE )
IF ( FT_CLIENT_INCLUDE_DIR )
    SET( FT_CLIENT_FOUND TRUE )
ENDIF ( FT_CLIENT_INCLUDE_DIR )

# Some status messages
IF ( FT_BUFFER_FOUND )
   IF ( NOT FT_BUFFER_FIND_QUIETLY )
       MESSAGE( STATUS "Found FT buffer: ${FT_BUFFER_LIBRARY_DIR} and include in ${FT_BUFFER_INCLUDE_DIR}" )
       MESSAGE( STATUS "      FT buffer: ${FT_BUFFER_LIBRARY}" )
   ENDIF()
ELSE ( FT_BUFFER_FOUND )
   IF (FT_BUFFER_FIND_REQUIRED)
      MESSAGE( FATAL_ERROR "FT_BUFFER not found!" )
   ENDIF()
ENDIF ( FT_BUFFER_FOUND )

IF ( FT_CLIENT_FOUND )
   IF ( NOT FT_CLIENT_FIND_QUIETLY )
       MESSAGE( STATUS "Found FT client: ${FT_CLIENT_INCLUDE_DIR}" )
   ENDIF()
ELSE ( FT_CLIENT_FOUND )
   IF (FT_CLIENT_FIND_REQUIRED)
      MESSAGE( FATAL_ERROR "FT_CLIENT not found!" )
   ENDIF()
ENDIF ( FT_CLIENT_FOUND )
