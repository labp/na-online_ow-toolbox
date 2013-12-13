//---------------------------------------------------------------------------
//
// Project: OpenWalnut ( http://www.openwalnut.org )
//
// Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS
// For more information see http://www.openwalnut.org/copying
//
// This file is part of OpenWalnut.
//
// OpenWalnut is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// OpenWalnut is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with OpenWalnut. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#include <core/common/WLogger.h>

#include "WLIOStatus.h"

std::string WLIOStatus::description( ioStatus_t statusCode )
{
    switch( statusCode )
    {
        case SUCCESS:
            return "I/O operation was successful.";
        case ERROR_FOPEN:
            return "Error while opening the file!";
        case ERROR_FREAD:
            return "Error while reading the file!";
        case ERROR_FWRITE:
            return "Error while writing the file!";
        case ERROR_UNKNOWN:
            return "Unknown error occurred!";
        case _USER_OFFSET:
            wlog::warn( "WLIOStatus" )
                            << "This is not a I/ status code - this constant should be user for user-defined status codes!";
            return "This is not a I/ status code - this constant should be user for user-defined status codes!";
        default:
            wlog::warn( "WLIOStatus" ) << "Unknown I/O status code!";
            return "Unknown I/O status code!";
    }
}

WLIOStatus::WLIOStatusInterpreter::WLIOStatusInterpreter()
{
}

WLIOStatus::WLIOStatusInterpreter::~WLIOStatusInterpreter()
{
}

std::string WLIOStatus::WLIOStatusInterpreter::getIOStatusDescription( WLIOStatus::ioStatus_t status )
{
    return WLIOStatus::description( status );
}