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

#include <string>

#include <core/common/WLogger.h>

#include "WLEMMCommandProcessor.h"

const std::string WLEMMCommandProcessor::CLASS = "WLEMMCommandProcessor";

WLEMMCommandProcessor::~WLEMMCommandProcessor()
{
}

bool WLEMMCommandProcessor::process( WLEMMCommand::SPtr cmdIn )
{
    bool succes;
    switch( cmdIn->getCommand() )
    {
        case WLEMMCommand::Command::COMPUTE:
            wlog::debug( CLASS ) << "Processing Command::COMPUTE";
            succes = processCompute( cmdIn->getEmm() );
            break;
        case WLEMMCommand::Command::INIT:
            wlog::debug( CLASS ) << "Processing Command::INIT";
            succes = processInit( cmdIn );
            break;
        case WLEMMCommand::Command::MISC:
            wlog::debug( CLASS ) << "Processing Command::MISC";
            succes = processMisc( cmdIn );
            break;
        case WLEMMCommand::Command::TIME_UPDATE:
            wlog::debug( CLASS ) << "Processing Command::TIME_UPDATE";
            succes = processTime( cmdIn );
            break;
        case WLEMMCommand::Command::RESET:
            wlog::debug( CLASS ) << "Processing Command::RESET";
            succes = processReset( cmdIn );
            break;
        default:
            wlog::error( CLASS ) << "Unknown Command::Enum!";
            succes = false;
    }
    if( !succes )
    {
        wlog::error( CLASS ) << "Error on processing command:\n" << *cmdIn;
    }
    return succes;
}
