#---------------------------------------------------------------------------
# This script searches the libraries and header files for FieldTrip Buffer on Ubuntu Linux.
#
# WWW: http://fieldtrip.fcdonders.nl
# Author: Christof Pieloth
#
# The following variables will be filled:
#   * FTB_BUFFER_FOUND - if FT_BUFFER header and library found
#   * FTB_CLIENT_FOUND - if FT_CLIENT header and library found
#   * FTB_BUFFER_INCLUDE_DIR - the path of FT_BUFFER header files
#   * FTB_CLIENT_INCLUDE_DIR - the path of FT_CLIENT header files
#   * FTB_BUFFER_LIBRARY - FT_BUFFER library
#   * FTB_CLIENT_LIBRARY - FT_BUFFER library
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
# FTB_BUFFER_INCLUDE_DIR
#---------------------------------------------------------------------------------------------------------------------------------

FIND_PATH( FTB_BUFFER_INCLUDE_DIR buffer.h HINTS 
        $ENV{FTB_BUFFER_INCLUDE_DIR} 
        $ENV{HOME}/na-online_dependencies/fieldtrip/realtime/src/buffer/src
        /opt/na-online_dependencies/fieldtrip/realtime/src/buffer/src
        /opt/include/fieldtrip/realtime/src/buffer/src
        /opt/include
        
)


#---------------------------------------------------------------------------------------------------------------------------------
# FTB_CLIENT_INCLUDE_DIR
#---------------------------------------------------------------------------------------------------------------------------------

FIND_PATH( FTB_CLIENT_INCLUDE_DIR FtBuffer.h HINTS 
        $ENV{FTB_CLIENT_INCLUDE_DIR}
        $ENV{HOME}/na-online_dependencies/fieldtrip/realtime/src/buffer/cpp
        /opt/na-online_dependencies/fieldtrip/realtime/src/buffer/cpp
        /opt/include/fieldtrip/realtime/src/buffer/cpp
        /opt/include
)


#---------------------------------------------------------------------------------------------------------------------------------
# FTB_BUFFER_LIBRARY
#---------------------------------------------------------------------------------------------------------------------------------

IF ( DEFINED ENV{FTB_BUFFER_LIBRARY} )
	SET(FTB_BUFFER_LIBRARY $ENV{FTB_BUFFER_LIBRARY})
ELSE ( DEFINED ENV{FTB_BUFFER_LIBRARY} )
	FIND_LIBRARY( FTB_BUFFER_LIBRARY libFtbBuffer.a HINTS
        $ENV{FTB_LIBRARY_DIR} 
        $ENV{HOME}/na-online_dependencies/fieldtrip/realtime/src/buffer/src
        /opt/na-online_dependencies/fieldtrip/realtime/src/buffer/src
        /opt/lib/fieldtrip
        /opt/lib
	)
ENDIF ( DEFINED ENV{FTB_BUFFER_LIBRARY} )


#---------------------------------------------------------------------------------------------------------------------------------
# FTB_CLIENT_LIBRARY
#---------------------------------------------------------------------------------------------------------------------------------

IF ( DEFINED ENV{FTB_CLIENT_LIBRARY} )
	SET(FTB_CLIENT_LIBRARY $ENV{FTB_CLIENT_LIBRARY})
ELSE ( DEFINED ENV{FTB_CLIENT_LIBRARY} )
	FIND_LIBRARY( FTB_CLIENT_LIBRARY libFtbClient.a HINTS
        $ENV{FTB_LIBRARY_DIR} 
        $ENV{HOME}/na-online_dependencies/fieldtrip/realtime/src/buffer/cpp
        /opt/na-online_dependencies/fieldtrip/realtime/src/buffer/cpp
        /opt/lib/fieldtrip
        /opt/lib
	)
ENDIF ( DEFINED ENV{FTB_CLIENT_LIBRARY} )


#---------------------------------------------------------------------------------------------------------------------------------
# Finalize setup
#---------------------------------------------------------------------------------------------------------------------------------

SET( FTB_BUFFER_FOUND FALSE )
IF ( FTB_BUFFER_INCLUDE_DIR AND FTB_BUFFER_LIBRARY )
    SET( FTB_BUFFER_FOUND TRUE )
ENDIF ( FTB_BUFFER_INCLUDE_DIR AND FTB_BUFFER_LIBRARY )

SET( FTB_CLIENT_FOUND FALSE )
IF ( FTB_CLIENT_INCLUDE_DIR AND FTB_CLIENT_LIBRARY )
    SET( FTB_CLIENT_FOUND TRUE )
ENDIF ( FTB_CLIENT_INCLUDE_DIR AND FTB_CLIENT_LIBRARY )

# Some status messages
IF ( FTB_BUFFER_FOUND )
   IF ( NOT FTB_BUFFER_FIND_QUIETLY )
       MESSAGE( STATUS "Found FTB buffer: ${FTB_BUFFER_INCLUDE_DIR}" )
       MESSAGE( STATUS "      FTB buffer: ${FTB_BUFFER_LIBRARY}" )
   ENDIF()
ELSE ( FTB_BUFFER_FOUND )
   IF (FTB_BUFFER_FIND_REQUIRED)
      MESSAGE( FATAL_ERROR "FT_BUFFER not found!" )
   ENDIF()
ENDIF ( FTB_BUFFER_FOUND )

IF ( FTB_CLIENT_FOUND )
   IF ( NOT FTB_CLIENT_FIND_QUIETLY )
       MESSAGE( STATUS "Found FTB client: ${FTB_CLIENT_INCLUDE_DIR}" )
       MESSAGE( STATUS "      FTB client: ${FTB_CLIENT_LIBRARY}" )
   ENDIF()
ELSE ( FTB_CLIENT_FOUND )
   IF (FTB_CLIENT_FIND_REQUIRED)
      MESSAGE( FATAL_ERROR "FT_CLIENT not found!" )
   ENDIF()
ENDIF ( FTB_CLIENT_FOUND )
