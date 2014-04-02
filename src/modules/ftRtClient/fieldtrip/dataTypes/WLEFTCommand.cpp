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

#include <core/common/WAssert.h>

#include <modules/ftRtClient/fieldtrip/dataTypes/WLEFTCommand.h>

WLEFTCommand::ContainerT WLEFTCommand::values()
{
    ContainerT commands;
    commands.insert( WLEFTCommand::UNSPECIFIED );
    commands.insert( WLEFTCommand::PUT_HDR );
    commands.insert( WLEFTCommand::PUT_DAT );
    commands.insert( WLEFTCommand::PUT_EVT );
    commands.insert( WLEFTCommand::PUT_OK );
    commands.insert( WLEFTCommand::PUT_ERR );
    commands.insert( WLEFTCommand::GET_HDR );
    commands.insert( WLEFTCommand::GET_DAT );
    commands.insert( WLEFTCommand::GET_EVT );
    commands.insert( WLEFTCommand::GET_OK );
    commands.insert( WLEFTCommand::GET_ERR );
    commands.insert( WLEFTCommand::FLUSH_HDR );
    commands.insert( WLEFTCommand::FLUSH_DAT );
    commands.insert( WLEFTCommand::FLUSH_EVT );
    commands.insert( WLEFTCommand::FLUSH_OK );
    commands.insert( WLEFTCommand::FLUSH_ERR );
    commands.insert( WLEFTCommand::WAIT_DAT );
    commands.insert( WLEFTCommand::WAIT_OK );
    commands.insert( WLEFTCommand::WAIT_ERR );
    return commands;
}

inline std::string WLEFTCommand::name( Enum val )
{
    switch( val )
    {
        case WLEFTCommand::UNSPECIFIED:
            return "Unspecified";
        case WLEFTCommand::PUT_HDR:
            return "Put Header";
        case WLEFTCommand::PUT_DAT:
            return "Put Data";
        case WLEFTCommand::PUT_EVT:
            return "Put Events";
        case WLEFTCommand::PUT_OK:
            return "Put OK";
        case WLEFTCommand::PUT_ERR:
            return "Put Error";
        case WLEFTCommand::GET_HDR:
            return "Get Header";
        case WLEFTCommand::GET_DAT:
            return "Get Data";
        case WLEFTCommand::GET_EVT:
            return "Get Events";
        case WLEFTCommand::GET_OK:
            return "Get OK";
        case WLEFTCommand::GET_ERR:
            return "Get Error";
        case WLEFTCommand::FLUSH_HDR:
            return "Flush Header";
        case WLEFTCommand::FLUSH_DAT:
            return "Flush Data";
        case WLEFTCommand::FLUSH_EVT:
            return "Flush Events";
        case WLEFTCommand::FLUSH_OK:
            return "Flush OK";
        case WLEFTCommand::FLUSH_ERR:
            return "Flush Error";
        case WLEFTCommand::WAIT_DAT:
            return "Wait Data";
        case WLEFTCommand::WAIT_OK:
            return "Wait OK";
        case WLEFTCommand::WAIT_ERR:
            return "Wait Error";
        default:
            WAssert( false, "Unknown WLEFTCommand!" );
            return WLEFTCommand::name( WLEFTCommand::UNSPECIFIED );
    }
}
