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

#include <string>

#include "WFTConnectionUnix.h"

const std::string WFTConnectionUnix::CLASS = "WFTConnectionUnix";

WFTConnectionUnix::WFTConnectionUnix( std::string pathname, int retry ) :
                WFTConnection::WFTConnection( retry ), m_pathname( pathname )
{
}

WFTConnectionUnix::~WFTConnectionUnix()
{
}

bool WFTConnectionUnix::connect()
{
    return connectUnix( m_pathname.c_str() );
}

std::string WFTConnectionUnix::getConnectionString() const
{
    return m_pathname;
}

std::string WFTConnectionUnix::getName() const
{
    return "Connection Unix";
}

const std::string WFTConnectionUnix::getPathName() const
{
    return m_pathname;
}

void WFTConnectionUnix::set( std::string pathname )
{
    if( isOpen() )
    {
        return;
    }

    m_pathname = pathname;
}
