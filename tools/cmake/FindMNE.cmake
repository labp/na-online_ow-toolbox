#---------------------------------------------------------------------------
# This script searches the MNE libraries and header files on Ubuntu Linux.
#
# WWW: https://github.com/mne-tools/mne-cpp
# Author: Christof Pieloth
#
# The following variables will be filled:
#   * MNE_FOUND - if MNE header and core library found
#   * MNE_INCLUDE_DIR - the path of MNE header files
#   * MNE_LIBRARY_DIR - the path to MNE libraries
#   * MNE_LIBRARY - MNE core library
#   * MNE_FIFF_LIBRARY
#   * MNE_FS_LIBRARY
#   * MNE_GENERICS_LIBRARY
#   * MNE_INVRT_LIBRARY
#   * MNE_UTILS_LIBRARY
#   * MNE_RTCLIENT_LIBRARY
#   * MNE_RTCOMMAND_LIBRARY
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# MNE_INCLUDE_DIR
#---------------------------------------------------------------------------

FIND_PATH( MNE_INCLUDE_DIR mne/mne.h HINTS 
        $ENV{MNE_INCLUDE_DIR} 
        /opt/include
        /opt/include/MNE
)



#---------------------------------------------------------------------------
# MNE_LIBRARY and MNE_LIBRARY_DIR
#---------------------------------------------------------------------------

FIND_LIBRARY( MNE_LIBRARY_RELEASE MNE1Mne HINTS
        $ENV{MNE_LIBRARY_DIR} 
        /opt/lib/MNE
        /opt/lib
)

FIND_LIBRARY( MNE_LIBRARY_DEBUG MNE1Mned HINTS
        $ENV{MNE_LIBRARY_DIR} 
        /opt/lib/MNE
        /opt/lib
)

# Comment/Uncomment these parts to use CMake's release/debug switch
#IF( CMAKE_BUILD_TYPE MATCHES Release )
	SET( MNE_LIBRARY ${MNE_LIBRARY_RELEASE} )
#ELSE()
#	SET( MNE_LIBRARY ${MNE_LIBRARY_DEBUG} )
#ENDIF()

# Retrieve library path for other MNE libraries
get_filename_component( MNE_LIBRARY_DIR ${MNE_LIBRARY} PATH )



#---------------------------------------------------------------------------
# Other libraries
#---------------------------------------------------------------------------

FIND_LIBRARY( MNE_FIFF_LIBRARY_RELEASE MNE1Fiff PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_FIFF_LIBRARY_DEBUG MNE1Fiffd PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_FS_LIBRARY_RELEASE MNE1Fs PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_FS_LIBRARY_DEBUG MNE1Fsd PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_GENERICS_LIBRARY_RELEASE MNE1Generics PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_GENERICS_LIBRARY_DEBUG MNE1Genericsd PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_INVRT_LIBRARY_RELEASE MNE1InvRt PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_INVRT_LIBRARY_DEBUG MNE1InvRtd PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_UTILS_LIBRARY_RELEASE MNE1Utils PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_UTILS_LIBRARY_DEBUG MNE1Utilsd PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_RTCLIENT_LIBRARY_RELEASE MNE1RtClient PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_RTCLIENT_LIBRARY_DEBUG MNE1RtClientd PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_RTCOMMAND_LIBRARY_RELEASE MNE1RtCommand PATHS ${MNE_LIBRARY_DIR} )
FIND_LIBRARY( MNE_RTCOMMAND_LIBRARY_DEBUG MNE1RtCommandd PATHS ${MNE_LIBRARY_DIR} )

# Comment/Uncomment these parts to use CMake's release/debug switch
#IF( CMAKE_BUILD_TYPE MATCHES Release )
	SET( MNE_FIFF_LIBRARY ${MNE_FIFF_LIBRARY_RELEASE} )
	SET( MNE_FS_LIBRARY ${MNE_FS_LIBRARY_RELEASE} )
	SET( MNE_GENERICS_LIBRARY ${MNE_GENERICS_LIBRARY_RELEASE} )
	SET( MNE_INVRT_LIBRARY ${MNE_INVRT_LIBRARY_RELEASE} )
	SET( MNE_UTILS_LIBRARY ${MNE_UTILS_LIBRARY_RELEASE} )
	SET( MNE_RTCLIENT_LIBRARY ${MNE_RTCLIENT_LIBRARY_RELEASE} )
	SET( MNE_RTCOMMAND_LIBRARY ${MNE_RTCOMMAND_LIBRARY_RELEASE} )
#ELSE()
#	SET( MNE_FIFF_LIBRARY ${MNE_FIFF_LIBRARY_DEBUG} )
#	SET( MNE_FS_LIBRARY ${MNE_FS_LIBRARY_DEBUG} )
#	SET( MNE_GENERICS_LIBRARY ${MNE_GENERICS_LIBRARY_DEBUG} )
#	SET( MNE_INVRT_LIBRARY ${MNE_INVRT_LIBRARY_DEBUG} )
#	SET( MNE_UTILS_LIBRARY ${MNE_UTILS_LIBRARY_DEBUG} )
#	SET( MNE_RTCLIENT_LIBRARY ${MNE_RTCLIENT_LIBRARY_DEBUG} )
#	SET( MNE_RTCOMMAND_LIBRARY ${MNE_RTCOMMAND_LIBRARY_DEBUG} )
#ENDIF()

#---------------------------------------------------------------------------
# Finalize setup
#---------------------------------------------------------------------------

SET( MNE_FOUND FALSE )
IF ( MNE_INCLUDE_DIR AND MNE_LIBRARY_DIR )
    SET( MNE_FOUND TRUE )
ENDIF ( MNE_INCLUDE_DIR AND MNE_LIBRARY_DIR )

# Some status messages
IF ( MNE_FOUND )
   IF ( NOT MNE_FIND_QUIETLY )
       MESSAGE( STATUS "Found MNE: ${MNE_LIBRARY_DIR} and include in ${MNE_INCLUDE_DIR}" )
       MESSAGE( STATUS "      MNE base: ${MNE_LIBRARY}" )
       MESSAGE( STATUS "      MNE fiff: ${MNE_FIFF_LIBRARY}" )
       MESSAGE( STATUS "      MNE fs: ${MNE_FS_LIBRARY}" )
       MESSAGE( STATUS "      MNE generics: ${MNE_GENERICS_LIBRARY}" )
       MESSAGE( STATUS "      MNE invrt: ${MNE_INVRT_LIBRARY}" )
       MESSAGE( STATUS "      MNE utils: ${MNE_UTILS_LIBRARY}" )
       MESSAGE( STATUS "      MNE rtClient: ${MNE_RTCLIENT_LIBRARY}" )
       MESSAGE( STATUS "      MNE rtCommand: ${MNE_RTCOMMAND_LIBRARY}" )
   ENDIF()
ELSE ( MNE_FOUND )
   IF (MNE_FIND_REQUIRED)
      MESSAGE( FATAL_ERROR "MNE not found!" )
   ENDIF()
ENDIF ( MNE_FOUND )
