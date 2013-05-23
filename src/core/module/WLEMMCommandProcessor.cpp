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

#include "WLEMMCommandProcessor.h"

const std::string WLEMMCommandProcessor::CLASS = "WLModuleChained";

WLEMMCommandProcessor::~WLEMMCommandProcessor()
{
}

bool WLEMMCommandProcessor::process( WLEMMCommand::SPtr labp )
{
    bool succes;
    switch( labp->getCommand() )
    {
        case WLEMMCommand::Command::COMPUTE:
            wlog::debug( CLASS ) << "Processing Command::COMPUTE";
            succes = processCompute( labp->getEmm() );
            break;
        case WLEMMCommand::Command::INIT:
            wlog::debug( CLASS ) << "Processing Command::INIT";
            succes = processInit( labp );
            break;
        case WLEMMCommand::Command::MISC:
            wlog::debug( CLASS ) << "Processing Command::MISC";
            succes = processMisc( labp );
            break;
        case WLEMMCommand::Command::RESET:
            wlog::debug( CLASS ) << "Processing Command::RESET";
            succes = processReset( labp );
            break;
        default:
            wlog::error( CLASS ) << "Unknown Command::Enum!";
            return false;
    }
    if( !succes )
    {
        wlog::error( CLASS ) << "Error on processing.";
    }
    return succes;
}
