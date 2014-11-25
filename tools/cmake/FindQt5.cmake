#---------------------------------------------------------------------------------------------------------------------------------
# This script searches for Qt5 libraries and header files on Ubuntu Linux.
#
# WWW: http://qt-project.org/downloads
# Author: Christof Pieloth
#
# The following variables will be filled:
#   * QT5_FOUND - if Qt5 header and core library found
#   * QT5_INCLUDE_DIR - the path of Qt5 header files
#   * QT5_LIBRARY_DIR - the path to Qt5 libraries
#   * QT5_QTCORE_INCLUDE_DIR - the path of Qt5Core header files
#   * QT5_QTCORE_LIBRARY - Qt5Core library
#   * QT5_QTNETWORK_INCLUDE_DIR - the path of Qt5Network header files
#   * QT5_QTNETWORK_LIBRARY - Qt5Network library
#   * QT5_QTCONCURRENT_INCLUDE_DIR - the path of Qt5Concurrent header files
#   * QT5_QTCONCURRENT_LIBRARY - Qt5Concurrent library
#
#---------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------------------
# QT5_INCLUDE_DIR
#---------------------------------------------------------------------------------------------------------------------------------

FIND_PATH( QT5_INCLUDE_DIR QtCore/QtCore HINTS 
        $ENV{QT5_INCLUDE_DIR}
        $ENV{HOME}/na-online_dependencies/qt5/include
        /opt/na-online_dependencies/qt5/include
        /opt/include/qt5
        /opt/include
)


#---------------------------------------------------------------------------------------------------------------------------------
# QTCORE - library and include path
#---------------------------------------------------------------------------------------------------------------------------------

FIND_LIBRARY( QT5_QTCORE_LIBRARY Qt5Core HINTS
	$ENV{QT5_LIBRARY_DIR}
	$ENV{HOME}/na-online_dependencies/qt5/lib
    /opt/na-online_dependencies/qt5/lib
	/opt/lib/qt5
	/opt/lib
)

# Retrieve library path for other Qt5 libraries
get_filename_component( QT5_LIBRARY_DIR ${QT5_QTCORE_LIBRARY} PATH )

# Set include dir
IF ( QT5_QTCORE_LIBRARY AND QT5_INCLUDE_DIR )
	SET ( QT5_QTCORE_INCLUDE_DIR ${QT5_INCLUDE_DIR}/QtCore)
ENDIF()


#---------------------------------------------------------------------------------------------------------------------------------
# QTNETWORK - library and include path
#---------------------------------------------------------------------------------------------------------------------------------

FIND_LIBRARY( QT5_QTNETWORK_LIBRARY Qt5Network HINTS
	$ENV{QT5_LIBRARY_DIR} 
	$ENV{HOME}/na-online_dependencies/qt5/lib
    /opt/na-online_dependencies/qt5/lib
	/opt/lib/qt5
	/opt/lib
)

# Set include dir
IF ( QT5_QTNETWORK_LIBRARY AND QT5_INCLUDE_DIR )
	SET ( QT5_QTNETWORK_INCLUDE_DIR ${QT5_INCLUDE_DIR}/QtNetwork)
ENDIF ( QT5_QTNETWORK_LIBRARY AND QT5_INCLUDE_DIR )


#---------------------------------------------------------------------------------------------------------------------------------
# QTCONCURRENT - library and include path
#---------------------------------------------------------------------------------------------------------------------------------

FIND_LIBRARY( QT5_QTCONCURRENT_LIBRARY Qt5Concurrent HINTS
	$ENV{QT5_LIBRARY_DIR} 
	$ENV{HOME}/na-online_dependencies/qt5/lib
    /opt/na-online_dependencies/qt5/lib
	/opt/lib/qt5
	/opt/lib
)

# Set include dir
IF ( QT5_QTCONCURRENT_LIBRARY AND QT5_INCLUDE_DIR )
	SET ( QT5_QTCONCURRENT_INCLUDE_DIR ${QT5_INCLUDE_DIR}/QtConcurrent)
ENDIF ( QT5_QTCONCURRENT_LIBRARY AND QT5_INCLUDE_DIR )


#---------------------------------------------------------------------------------------------------------------------------------
# Finalize setup
#---------------------------------------------------------------------------------------------------------------------------------

# Set QT5_FOUND
SET( QT5_FOUND FALSE )
IF ( QT5_INCLUDE_DIR AND QT5_LIBRARY_DIR )
    SET( QT5_FOUND TRUE )
ENDIF ( QT5_INCLUDE_DIR AND QT5_LIBRARY_DIR )

# Some status messages
IF ( QT5_FOUND )
   IF ( NOT QT5_FIND_QUIETLY )
       MESSAGE( STATUS "Found Qt5: ${QT5_LIBRARY_DIR} and include in ${QT5_INCLUDE_DIR}" )
       MESSAGE( STATUS "      Qt5 core: ${QT5_QTCORE_INCLUDE_DIR}" )
       MESSAGE( STATUS "      Qt5 core: ${QT5_QTCORE_LIBRARY}" )
       MESSAGE( STATUS "      Qt5 network: ${QT5_QTNETWORK_INCLUDE_DIR}" )
       MESSAGE( STATUS "      Qt5 network: ${QT5_QTNETWORK_LIBRARY}" )
       MESSAGE( STATUS "      Qt5 concurrent: ${QT5_QTCONCURRENT_INCLUDE_DIR}" )
       MESSAGE( STATUS "      Qt5 concurrent: ${QT5_QTCONCURRENT_LIBRARY}" )
   ENDIF()
ELSE ( QT5_FOUND )
   IF (QT5_FIND_REQUIRED)
      MESSAGE( FATAL_ERROR "Qt5 not found!" )
   ENDIF()
ENDIF ( QT5_FOUND )
