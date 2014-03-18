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

#include "io/request/WFTRequest_GetData.h"
#include "io/request/WFTRequest_GetHeader.h"
#include "io/request/WFTRequest_WaitData.h"
#include "io/response/WFTResponse.h"
#include "io/dataTypes/WFTHeader.h"
#include "io/dataTypes/WFTChunk.h"
#include "WFTRtClient.h"

#include <FtBuffer.h>

const std::string WFTRtClient::CLASS = "WFTRtClient";

const UINT32_T WFTRtClient::DEFAULT_WAIT_TIMEOUT = 40;

WFTRtClient::WFTRtClient() :
                m_waitTimeout_ms( DEFAULT_WAIT_TIMEOUT ), m_samples( 0 ), m_events( 0 )
{
    m_reqBuilder.reset( new WFTRequestBuilder );
    m_ftHeader.reset( new WFTHeader );
}

WFTRtClient::~WFTRtClient()
{

}

void WFTRtClient::resetClient()
{
    m_samples = 0;
    m_events = 0;
    m_svr_samp_evt.nsamples = 0;
    m_svr_samp_evt.nevents = 0, m_waitTimeout_ms = DEFAULT_WAIT_TIMEOUT;

    if( isConnected() )
    {
        disconnect();
    }
}

WFTConnection::SPtr WFTRtClient::getConnection() const
{
    return m_connection;
}

WFTHeader::SPtr WFTRtClient::getHeader() const
{
    return m_ftHeader;
}

WFTData::SPtr WFTRtClient::getData() const
{
    return m_ftData;
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

bool WFTRtClient::hasNewSamples() const
{
    return m_svr_samp_evt.nsamples > m_samples;
}

bool WFTRtClient::hasNewEvents() const
{
    return m_svr_samp_evt.nevents > m_events;
}

UINT32_T WFTRtClient::getSampleCount() const
{
    return m_samples;
}

UINT32_T WFTRtClient::getEventCount() const
{
    return m_events;
}

bool WFTRtClient::doRequest( WFTRequest& request, WFTResponse& response )
{
    if( !isConnected() )
    {
        return false;
    }

    if( tcprequest( m_connection->getSocket(), request.out(), response.in() ) < 0 )
    {
        wlog::error( CLASS ) << "Error in communication - check buffer server.";
        return false;
    }

    return true;
}

bool WFTRtClient::doHeaderRequest()
{
    WFTResponse::SPtr response( new WFTResponse );
    WFTRequest_GetHeader::SPtr request = m_reqBuilder->buildRequest_GET_HDR();
    m_ftHeader.reset( new WFTHeader );

    if( !doRequest( *request, *response ) )
    {
        return false;
    }

    if( !m_ftHeader->parseResponse( response ) )
    {
        wlog::error( CLASS ) << "Error while parsing server response.";
        return false;
    }

    // check for new samples & events
    m_svr_samp_evt.nsamples = m_ftHeader->getHeaderDef().nsamples;
    m_svr_samp_evt.nevents = m_ftHeader->getHeaderDef().nevents;

    return true;
}

bool WFTRtClient::doWaitRequest( unsigned int samples, unsigned int events )
{
    WFTResponse::SPtr response( new WFTResponse );
    WFTRequest_WaitData::SPtr request = m_reqBuilder->buildRequest_WAIT_DAT( samples, events, m_waitTimeout_ms );

    if( !doRequest( *request, *response ) )
    {
        return false;
    }

    if( !response->checkWait( m_svr_samp_evt.nsamples, m_svr_samp_evt.nevents ) )
    {
        return false;
    }

    // do header request after flush/restart on server (server.samples < client.samples)
    if( m_svr_samp_evt.nsamples < samples )
    {
        if( !doHeaderRequest() )
        {
            return false;
        }
        m_samples = 0;
    }

    return true;
}

bool WFTRtClient::getNewSamples()
{
    if( m_ftHeader == 0 )
    {
        return false;
    }

    if( !hasNewSamples() )
    {
        return false;
    }

    WFTRequest_GetData::SPtr request = m_reqBuilder->buildRequest_GET_DAT( m_samples, m_svr_samp_evt.nsamples - 1 );
    WFTResponse::SPtr response( new WFTResponse );

    if( !doRequest( *request, *response ) )
    {
        return false;
    }

    m_ftData.reset( new WFTData );

    if( !m_ftData->parseResponse( response ) )
    {
        return false;
    }

    m_samples = m_svr_samp_evt.nsamples; // update number of read samples.

    return true;
}

bool WFTRtClient::getNewEvents()
{
    if( m_ftHeader == 0 )
    {
        return false;
    }

    if( !hasNewEvents() )
    {
        return false;
    }

    m_events = m_svr_samp_evt.nevents;

    return true;
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

UINT32_T WFTRtClient::getTimeout() const
{
    return m_waitTimeout_ms;
}

void WFTRtClient::setTimeout(UINT32_T timeout)
{
    m_waitTimeout_ms = timeout;
}
