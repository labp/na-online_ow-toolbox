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

#ifndef WLEMMCOMMANDPROCESSOR_H_
#define WLEMMCOMMANDPROCESSOR_H_

#include <string>

#include <core/kernel/WModule.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"

/**
 * Template for a module to process a received command.
 *
 * \author pieloth
 * \ingroup module
 */
class WLEMMCommandProcessor
{
public:
    static const std::string CLASS;

    virtual ~WLEMMCommandProcessor();

    /**
     * \brief Processes a command.
     * Processes the command by calling the specific method for WLEMMCommand::Command::Enum,
     * e.g. WLEMMCommandProcessor::processCompute.
     *
     * \param cmdIn Command to process.
     * \return True if processing was successful.
     */
    bool process( WLEMMCommand::SPtr cmdIn );

protected:
    virtual bool processCompute( WLEMMeasurement::SPtr emm ) = 0;
    virtual bool processInit( WLEMMCommand::SPtr cmdIn ) = 0;
    virtual bool processMisc( WLEMMCommand::SPtr cmdIn ) = 0;
    virtual bool processTime( WLEMMCommand::SPtr cmdIn ) = 0;
    virtual bool processReset( WLEMMCommand::SPtr cmdIn ) = 0;
};

#endif  // WLEMMCOMMANDPROCESSOR_H_
