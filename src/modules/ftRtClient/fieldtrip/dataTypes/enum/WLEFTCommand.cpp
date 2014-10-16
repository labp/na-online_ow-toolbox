//---------------------------------------------------------------------------
//
// Project: NA-Online ( http://www.labp.htwk-leipzig.de )
//
// Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
//
// This file is part of NA-Online.
//
// NA-Online is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NA-Online is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#include <core/common/WAssert.h>

#include "WLEFTCommand.h"

WLEFTCommand::ContainerT WLEFTCommand::values()
{
    ContainerT commands;
    commands.insert( WLEFTCommand::UNSPECIFIED );
    commands.insert( WLEFTCommand::PUTHDR );
    commands.insert( WLEFTCommand::PUTDAT );
    commands.insert( WLEFTCommand::PUTEVT );
    commands.insert( WLEFTCommand::PUTOK );
    commands.insert( WLEFTCommand::PUTERR );
    commands.insert( WLEFTCommand::GETHDR );
    commands.insert( WLEFTCommand::GETDAT );
    commands.insert( WLEFTCommand::GETEVT );
    commands.insert( WLEFTCommand::GETOK );
    commands.insert( WLEFTCommand::GETERR );
    commands.insert( WLEFTCommand::FLUSHHDR );
    commands.insert( WLEFTCommand::FLUSHDAT );
    commands.insert( WLEFTCommand::FLUSHEVT );
    commands.insert( WLEFTCommand::FLUSHOK );
    commands.insert( WLEFTCommand::FLUSHERR );
    commands.insert( WLEFTCommand::WAITDAT );
    commands.insert( WLEFTCommand::WAITOK );
    commands.insert( WLEFTCommand::WAITERR );
    return commands;
}

std::string WLEFTCommand::name( WLEFTCommand::Enum val )
{
    switch( val )
    {
        case WLEFTCommand::UNSPECIFIED:
            return "Unspecified";
        case WLEFTCommand::PUTHDR:
            return "Put Header";
        case WLEFTCommand::PUTDAT:
            return "Put Data";
        case WLEFTCommand::PUTEVT:
            return "Put Events";
        case WLEFTCommand::PUTOK:
            return "Put OK";
        case WLEFTCommand::PUTERR:
            return "Put Error";
        case WLEFTCommand::GETHDR:
            return "Get Header";
        case WLEFTCommand::GETDAT:
            return "Get Data";
        case WLEFTCommand::GETEVT:
            return "Get Events";
        case WLEFTCommand::GETOK:
            return "Get OK";
        case WLEFTCommand::GETERR:
            return "Get Error";
        case WLEFTCommand::FLUSHHDR:
            return "Flush Header";
        case WLEFTCommand::FLUSHDAT:
            return "Flush Data";
        case WLEFTCommand::FLUSHEVT:
            return "Flush Events";
        case WLEFTCommand::FLUSHOK:
            return "Flush OK";
        case WLEFTCommand::FLUSHERR:
            return "Flush Error";
        case WLEFTCommand::WAITDAT:
            return "Wait Data";
        case WLEFTCommand::WAITOK:
            return "Wait OK";
        case WLEFTCommand::WAITERR:
            return "Wait Error";
        default:
            WAssert( false, "Unknown WLEFTCommand!" );
            return WLEFTCommand::name( WLEFTCommand::UNSPECIFIED );
    }
}
