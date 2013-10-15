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

#ifndef WLEMMCOMMANDPROCESSOR_H_
#define WLEMMCOMMANDPROCESSOR_H_

#include <string>

#include <core/kernel/WModule.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"

class WLEMMCommandProcessor
{
public:
    static const std::string CLASS;

    virtual ~WLEMMCommandProcessor();

    bool process( WLEMMCommand::SPtr cmdIn );

protected:
    virtual bool processCompute( WLEMMeasurement::SPtr emm ) = 0;
    virtual bool processInit( WLEMMCommand::SPtr cmdIn ) = 0;
    virtual bool processMisc( WLEMMCommand::SPtr cmdIn ) = 0;
    virtual bool processTime( WLEMMCommand::SPtr cmdIn ) = 0;
    virtual bool processReset( WLEMMCommand::SPtr cmdIn ) = 0;
};

#endif  // WLEMMCOMMANDPROCESSOR_H_
