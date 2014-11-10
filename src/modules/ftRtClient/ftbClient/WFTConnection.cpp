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

#include <core/common/WLogger.h>

#include "WFTConnection.h"

const std::string WFTConnection::CLASS = "WFTConnection";

static const int NO_PORT = -1;

WFTConnection::WFTConnection( int retry ) :
                FtConnection::FtConnection( retry ), m_port( NO_PORT )
{
}

WFTConnection::~WFTConnection()
{
}

bool WFTConnection::connect()
{
    if( !m_host.empty() && m_port != NO_PORT )
    {
        return connectTcp( m_host.c_str(), m_port );
    }
    if( !m_path.empty() )
    {
        return connectUnix( m_path.c_str() );
    }

    wlog::error( CLASS ) << "Could not connect! Host, port or path is missing.";
    return false;

}

bool WFTConnection::connect( const std::string& address )
{
    return FtConnection::connect( address.c_str() );
}

std::string WFTConnection::getHost() const
{
    return m_host;
}

void WFTConnection::setHost( const std::string& host )
{
    m_host = host;
    m_path.clear();
}

int WFTConnection::getPort() const
{
    return m_port;
}

void WFTConnection::setPort( int port )
{
    m_path.clear();
    m_port = port;
}

std::string WFTConnection::getPath() const
{
    return m_path;
}

void WFTConnection::setPath( const std::string& path )
{

    m_host.clear();
    m_port = NO_PORT;
    m_path = path;
}

