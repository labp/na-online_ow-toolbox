#---------------------------------------------------------------------------
# This script searches for Qt5 static libraries and header files on Ubuntu Linux.
#
# WWW: http://qt-project.org/downloads
# Author: Christof Pieloth
#
# The following variables will be filled:
#   * QT5_FOUND - if Qt5 header and core library found
#   * QT5_INCLUDE_DIR - the path of Qt5 header files
#   * QT5_STATIC_LIBRARY_DIR - the path to Qt5 static libraries
#   * QT5_QTCORE_INCLUDE_DIR - the path of Qt5Core header files
#   * QT5_STATIC_QTCORE_LIBRARY - Qt5Core static library
#   * QT5_QTNETWORK_INCLUDE_DIR - the path of Qt5Network header files
#   * QT5_STATIC_QTNETWORK_LIBRARY - Qt5Network static library
#
# NOTE: FIND_LIBRARY is looking for *.so and *.a files!
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# QT5_INCLUDE_DIR
#---------------------------------------------------------------------------

FIND_PATH( QT5_INCLUDE_DIR_RELEASE QtCore/QtCore HINTS 
        $ENV{QT5_INCLUDE_DIR} 
        /opt/include/qt5
        /opt/include
)

FIND_PATH( QT5_INCLUDE_DIR_DEBUG QtCore/QtCore HINTS 
        $ENV{QT5_INCLUDE_DIR} 
        /opt/include/qt5d
)

IF( CMAKE_BUILD_TYPE MATCHES Release )
	SET( QT5_INCLUDE_DIR ${QT5_INCLUDE_DIR_RELEASE} )
ELSE()
	SET( QT5_INCLUDE_DIR ${QT5_INCLUDE_DIR_DEBUG} )
ENDIF()

#---------------------------------------------------------------------------
# QTCORE - static library and include path
#---------------------------------------------------------------------------

FIND_LIBRARY( QT5_STATIC_QTCORE_LIBRARY_RELEASE Qt5Core HINTS
	$ENV{QT5_STATIC_LIBRARY_DIR} 
	/opt/lib/qt5_static
	/opt/lib
)

FIND_LIBRARY( QT5_STATIC_QTCORE_LIBRARY_DEBUG Qt5Core HINTS
	$ENV{QT5_STATIC_LIBRARY_DIR} 
	/opt/lib/qt5_staticd
	/opt/lib
)

IF( CMAKE_BUILD_TYPE MATCHES Release )
	SET( QT5_STATIC_QTCORE_LIBRARY ${QT5_STATIC_QTCORE_LIBRARY_RELEASE} )
ELSE()
	SET( QT5_STATIC_QTCORE_LIBRARY ${QT5_STATIC_QTCORE_LIBRARY_DEBUG} )
ENDIF()


# Retrieve library path for other Qt5 libraries
get_filename_component( QT5_STATIC_LIBRARY_DIR ${QT5_STATIC_QTCORE_LIBRARY} PATH )

# Set include dir
IF ( QT5_STATIC_QTCORE_LIBRARY AND QT5_INCLUDE_DIR )
	SET ( QT5_QTCORE_INCLUDE_DIR ${QT5_INCLUDE_DIR}/QtCore)
ENDIF()



#---------------------------------------------------------------------------
# QTNETWORK - static library and include path
#---------------------------------------------------------------------------

FIND_LIBRARY( QT5_STATIC_QTNETWORK_LIBRARY_RELEASE Qt5Network HINTS
	$ENV{QT5_STATIC_LIBRARY_DIR} 
	/opt/lib/qt5_static
	/opt/lib
)

FIND_LIBRARY( QT5_STATIC_QTNETWORK_LIBRARY_DEBUG Qt5Network HINTS
	$ENV{QT5_STATIC_LIBRARY_DIR} 
	/opt/lib/qt5_staticd
	/opt/lib
)

IF( CMAKE_BUILD_TYPE MATCHES Release )
	SET( QT5_STATIC_QTNETWORK_LIBRARY ${QT5_STATIC_QTNETWORK_LIBRARY_RELEASE} )
ELSE()
	SET( QT5_STATIC_QTNETWORK_LIBRARY ${QT5_STATIC_QTNETWORK_LIBRARY_DEBUG} )
ENDIF()

# Set include dir
IF ( QT5_STATIC_QTNETWORK_LIBRARY AND QT5_INCLUDE_DIR )
	SET ( QT5_QTNETWORK_INCLUDE_DIR ${QT5_INCLUDE_DIR}/QtNetwork)
ENDIF ( QT5_STATIC_QTNETWORK_LIBRARY AND QT5_INCLUDE_DIR )



#---------------------------------------------------------------------------
# Finalize setup
#---------------------------------------------------------------------------

# Set QT5_FOUND
SET( QT5_FOUND FALSE )
IF ( QT5_INCLUDE_DIR AND QT5_STATIC_LIBRARY_DIR )
    SET( QT5_FOUND TRUE )
ENDIF ( QT5_INCLUDE_DIR AND QT5_STATIC_LIBRARY_DIR )

# Some status messages
IF ( QT5_FOUND )
   IF ( NOT QT5_FIND_QUIETLY )
       MESSAGE( STATUS "Found Qt5: ${QT5_STATIC_LIBRARY_DIR} and include in ${QT5_INCLUDE_DIR}" )
       MESSAGE( STATUS "      Qt5 core: ${QT5_QTCORE_INCLUDE_DIR}" )
       MESSAGE( STATUS "      Qt5 core: ${QT5_STATIC_QTCORE_LIBRARY}" )
       MESSAGE( STATUS "      Qt5 network: ${QT5_QTNETWORK_INCLUDE_DIR}" )
       MESSAGE( STATUS "      Qt5 network: ${QT5_STATIC_QTNETWORK_LIBRARY}" )
       MESSAGE( STATUS "      Qt5 NOTE: Please check if the found libraries are static and release/debug!" )
   ENDIF()
ELSE ( QT5_FOUND )
   IF (QT5_FIND_REQUIRED)
      MESSAGE( FATAL_ERROR "Qt5 not found!" )
   ENDIF()
ENDIF ( QT5_FOUND )
