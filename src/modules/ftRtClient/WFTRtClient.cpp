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

#include <buffer.h>

#include "core/common/WLogger.h"

#include "io/request/WFTRequest_GetHeader.h"
#include "io/response/WFTResponse.h"
#include "io/dataTypes/WFTHeader.h"
#include "io/dataTypes/WFTChunk.h"
#include "WFTRtClient.h"

#include <FtBuffer.h>

const std::string WFTRtClient::CLASS = "WFTRtClient";

WFTRtClient::WFTRtClient()
{
    m_reqBuilder.reset( new WFTRequestBuilder );
    m_ftHeader.reset( new WFTHeader );
}

WFTConnection::SPtr WFTRtClient::getConnection() const
{
    return m_connection;
}

WFTHeader::SPtr WFTRtClient::getHeader() const
{
    return m_ftHeader;
}

void WFTRtClient::setConnection( WFTConnection::SPtr connection )
{
    m_connection = connection;
}

bool WFTRtClient::connect()
{
    return m_connection == 0 ? false : m_connection->connect();
}

void WFTRtClient::disconnect()
{
    m_connection->disconnect();
}

bool WFTRtClient::isConnected()
{
    return m_connection == 0 ? false : m_connection->isOpen();
}

template< typename T >
bool WFTRtClient::getResponseAs( boost::shared_ptr< T > object, WFTResponse::SPtr response )
{
    if( !( boost::is_base_of< WFTRequestableObject, T >::value ) )
    {
        return false;
    }

    WFTRequestableObject& ptr = *object;

    return ptr.parseResponse( response );
}

bool WFTRtClient::doHeaderRequest()
{
    WFTResponse::SPtr response( new WFTResponse );
    m_ftHeader.reset( new WFTHeader );

    if( tcprequest( m_connection->getSocket(), m_reqBuilder->buildRequest_GET_HDR()->out(), response->in() ) < 0 )
    {
        wlog::error( CLASS ) << "Error in communication - check buffer server.";
        return false;
    }

    if( !m_ftHeader->parseResponse( response ) )
    {
        wlog::error( CLASS ) << "Error while parsing server response.";
        return false;
    }

    return true;
}
