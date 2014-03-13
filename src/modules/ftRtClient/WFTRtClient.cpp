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
}

WFTConnection::SPtr WFTRtClient::getConnection() const
{
    return m_connection;
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
    return m_connection->isOpen();
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

void WFTRtClient::doReqest()
{
    wlog::debug( CLASS ) << "doRequest()";
    /*
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
     */

    /* example for creating and retrieving a request.
     std::string a = "hello World";

     // create request
     FtBufferRequest request;
     request.prepPutHeader( 10, 2, 1 );
     request.prepPutHeaderAddChunk( 1, a.length(), a.c_str() );

     // get request message
     message_t mess = *request.out();
     headerdef_t *header = (headerdef_t*)mess.buf; // get message content as header (headerdef_t)

     wlog::debug( CLASS ) << "Message buffer size: " << mess.def->bufsize; // message content size
     wlog::debug( CLASS ) << "General header size: " << sizeof(headerdef_t);
     wlog::debug( CLASS ) << "header buffer size: " << header->bufsize; // size for chunks after headerdef

     ft_chunkdef_t *chunkdef = (ft_chunkdef_t*) mess.buf + sizeof(headerdef_t); // get first chunks def

     wlog::debug( CLASS ) << "chunk buffer size: " << chunkdef->size;

     int size = header->bufsize - sizeof(ft_chunkdef_t);
     char *str = (char*)malloc(size);
     memcpy(str, mess.buf + sizeof(headerdef_t) + sizeof(ft_chunkdef_t), size);
     wlog::debug( CLASS ) << "My String: " << str;
     */
}
