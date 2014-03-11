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
#include "WFTRtClient.h"

const std::string WFTRtClient::CLASS = "WFTRtClient";

WFTRtClient::WFTRtClient( WFTConnection::SPtr connection ) :
                m_connection( connection )
{
    m_reqBuilder.reset( new WFTRequestBuilder );
}

WFTConnection::SPtr WFTRtClient::getConnection() const
{
    return m_connection;
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
    return m_connection->isOpen();
}

template< typename T >
bool WFTRtClient::getResponseAs( boost::shared_ptr< T > object, WFTResponse::SPtr response )
{
    if( !( boost::is_base_of< WFTObject, T >::value ) )
    {
        return false;
    }

    WFTObject& ptr = *object;

    return ptr.parseResponse( response );
}

void WFTRtClient::doReqest()
{
    wlog::debug( CLASS ) << "doRequest()";

    WFTRequest_GetHeader::SPtr request = m_reqBuilder->buildRequest_GET_HDR();
    WFTResponse::SPtr response( new WFTResponse );

    if( tcprequest( m_connection->getSocket(), request->out(), response->in() ) < 0 )
    {
        wlog::debug( CLASS ) << "Fehler bei Request aufgetreten";
        return;
    }
    else
    {
        wlog::debug( CLASS ) << "Request erfolgreich";
    }

    WFTHeader::SPtr header( new WFTHeader );

    if( !getResponseAs< WFTHeader >( header, response ) )
    {
        wlog::debug( CLASS ) << "Umwandeln in Zieltyp gescheitert";
        return;
    }

    wlog::debug( CLASS ) << "Umwandeln in Zieltyp erfolgreich";

    wlog::debug( CLASS ) << "Samples: " << header->getHeaderDef().nsamples;
    wlog::debug( CLASS ) << "Channels: " << header->getHeaderDef().nchans;
    wlog::debug( CLASS ) << "Events: " << header->getHeaderDef().nevents;
    wlog::debug( CLASS ) << "Sample Frequency: " << header->getHeaderDef().fsample;
    wlog::debug( CLASS ) << "Data Type: " << header->getHeaderDef().data_type;
    wlog::debug( CLASS ) << "Buffer Size: " << header->getHeaderDef().bufsize;

}
