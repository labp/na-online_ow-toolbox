#------------------------------------------------------------------------------------------------------------------------------------------
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
#   * QT5_QTCONCURRENT_INCLUDE_DIR - the path of Qt5Concurrent header files
#   * QT5_STATIC_QTCONCURRENT_LIBRARY - Qt5Concurrent static library
#   * QT5_STATIC_DEPENDENCIES - Library dependencies for static Qt5
#
# NOTE: FIND_LIBRARY is looking for *.so and *.a files!
#---------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
# QT5_STATIC_DEPENDENCIES
#---------------------------------------------------------------------------------------------------------------------------------

SET(QT5_STATIC_DEPENDENCIES pthread rt dl)


#---------------------------------------------------------------------------------------------------------------------------------
# QT5_INCLUDE_DIR
#---------------------------------------------------------------------------------------------------------------------------------

FIND_PATH( QT5_INCLUDE_DIR QtCore/QtCore HINTS 
        $ENV{QT5_INCLUDE_DIR}
        $ENV{HOME}/na-online_dependencies/qt5_static/include
        /opt/na-online_dependencies/qt5_static/include
        /opt/include/qt5
        /opt/include
)


#---------------------------------------------------------------------------------------------------------------------------------
# QTCORE - static library and include path
#---------------------------------------------------------------------------------------------------------------------------------

FIND_LIBRARY( QT5_STATIC_QTCORE_LIBRARY Qt5Core HINTS
	$ENV{QT5_STATIC_LIBRARY_DIR}
	$ENV{HOME}/na-online_dependencies/qt5_static/lib
    /opt/na-online_dependencies/qt5_static/lib
	/opt/lib/qt5_static
	/opt/lib
)

# Retrieve library path for other Qt5 libraries
get_filename_component( QT5_STATIC_LIBRARY_DIR ${QT5_STATIC_QTCORE_LIBRARY} PATH )

# Set include dir
IF ( QT5_STATIC_QTCORE_LIBRARY AND QT5_INCLUDE_DIR )
	SET ( QT5_QTCORE_INCLUDE_DIR ${QT5_INCLUDE_DIR}/QtCore)
ENDIF()


#---------------------------------------------------------------------------------------------------------------------------------
# QTNETWORK - static library and include path
#---------------------------------------------------------------------------------------------------------------------------------

FIND_LIBRARY( QT5_STATIC_QTNETWORK_LIBRARY Qt5Network HINTS
	$ENV{QT5_STATIC_LIBRARY_DIR} 
	$ENV{HOME}/na-online_dependencies/qt5_static/lib
    /opt/na-online_dependencies/qt5_static/lib
	/opt/lib/qt5_static
	/opt/lib
)

# Set include dir
IF ( QT5_STATIC_QTNETWORK_LIBRARY AND QT5_INCLUDE_DIR )
	SET ( QT5_QTNETWORK_INCLUDE_DIR ${QT5_INCLUDE_DIR}/QtNetwork)
ENDIF ( QT5_STATIC_QTNETWORK_LIBRARY AND QT5_INCLUDE_DIR )


#---------------------------------------------------------------------------------------------------------------------------------
# QTCONCURRENT - static library and include path
#---------------------------------------------------------------------------------------------------------------------------------

FIND_LIBRARY( QT5_STATIC_QTCONCURRENT_LIBRARY Qt5Concurrent HINTS
	$ENV{QT5_STATIC_LIBRARY_DIR} 
	$ENV{HOME}/na-online_dependencies/qt5_static/lib
    /opt/na-online_dependencies/qt5_static/lib
	/opt/lib/qt5_static
	/opt/lib
)

# Set include dir
IF ( QT5_STATIC_QTCONCURRENT_LIBRARY AND QT5_INCLUDE_DIR )
	SET ( QT5_QTCONCURRENT_INCLUDE_DIR ${QT5_INCLUDE_DIR}/QtConcurrent)
ENDIF ( QT5_STATIC_QTCONCURRENT_LIBRARY AND QT5_INCLUDE_DIR )


#---------------------------------------------------------------------------------------------------------------------------------
# Finalize setup
#---------------------------------------------------------------------------------------------------------------------------------

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
       MESSAGE( STATUS "      Qt5 concurrent: ${QT5_QTCONCURRENT_INCLUDE_DIR}" )
       MESSAGE( STATUS "      Qt5 concurrent: ${QT5_STATIC_QTCONCURRENT_LIBRARY}" )
       MESSAGE( STATUS "      Qt5 NOTE: Please check if the found libraries are static and release/debug!" )
   ENDIF()
ELSE ( QT5_FOUND )
   IF (QT5_FIND_REQUIRED)
      MESSAGE( FATAL_ERROR "Qt5 not found!" )
   ENDIF()
ENDIF ( QT5_FOUND )
