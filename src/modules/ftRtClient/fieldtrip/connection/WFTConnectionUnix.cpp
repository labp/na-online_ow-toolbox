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
