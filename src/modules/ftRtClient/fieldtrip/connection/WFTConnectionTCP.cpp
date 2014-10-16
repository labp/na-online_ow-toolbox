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

#include "boost/lexical_cast.hpp"

#include "WFTConnectionTCP.h"

const std::string WFTConnectionTCP::CLASS = "WFTConnectionTCP";

WFTConnectionTCP::WFTConnectionTCP( std::string host, int port, int retry ) :
                WFTConnection::WFTConnection( retry ), m_host( host ), m_port( port )
{

}

WFTConnectionTCP::~WFTConnectionTCP()
{

}

bool WFTConnectionTCP::connect()
{
    return connectTcp( m_host.c_str(), m_port );
}

std::string WFTConnectionTCP::getConnectionString() const
{
    return m_host + ":" + boost::lexical_cast< std::string >( m_port );
}

std::string WFTConnectionTCP::getName() const
{
    return "Connection TCP";
}

const std::string WFTConnectionTCP::getHost() const
{
    return m_host;
}

int WFTConnectionTCP::getPort() const
{
    return m_port;
}

void WFTConnectionTCP::set( std::string host, int port )
{
    if( isOpen() )
    {
        return;
    }

    m_host = host;
    m_port = port;
}
