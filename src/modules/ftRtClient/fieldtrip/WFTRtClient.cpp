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

#include <boost/thread/strict_lock.hpp>

#include <buffer.h>
#include <FtBuffer.h>

#include "core/common/WLogger.h"

#include "io/request/WFTRequest.h"
#include "io/request/WFTRequest_GetData.h"
#include "io/request/WFTRequest_GetEvent.h"
#include "io/request/WFTRequest_GetHeader.h"
#include "io/request/WFTRequest_WaitData.h"
#include "io/response/WFTResponse.h"
#include "dataTypes/WFTHeader.h"
#include "dataTypes/WFTChunk.h"
#include "WFTEventIterator.h"
#include "WFTRtClient.h"

const std::string WFTRtClient::CLASS = "WFTRtClient";

const UINT32_T WFTRtClient::DEFAULT_WAIT_TIMEOUT = 40;

WFTRtClient::WFTRtClient() :
                m_waitTimeout_ms( DEFAULT_WAIT_TIMEOUT ), m_samples( 0 ), m_events( 0 )
{
    m_reqBuilder.reset( new WFTRequestBuilder );
    m_ftHeader.reset( new WFTHeader( 0, 0, 0 ) );
    m_ftEvents.reset( new WFTEventList );
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

WFTEventList::SPtr WFTRtClient::getEventList() const
{
    return m_ftEvents;
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
    // Lock the client for thread-safe requests.
    boost::strict_lock< WFTRtClient > guard( *this );

    if( !isConnected() )
    {
        wlog::error( CLASS ) << "The client is not connected.";

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
    m_ftHeader.reset( new WFTHeader( 0, 0, 0 ) );

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
        wlog::error( CLASS ) << "Error while doing Wait-Request.";

        return false;
    }

    if( !response->checkWait( m_svr_samp_evt.nsamples, m_svr_samp_evt.nevents ) )
    {
        wlog::error( CLASS ) << "Error while checking Wait-Request response.";

        wlog::error( CLASS ) << *response;
        return false;
    }

    // do header request after flush/restart on server (server.samples < client.samples)
    if( m_svr_samp_evt.nsamples < samples || m_svr_samp_evt.nevents < events )
    {
        if( !doHeaderRequest() )
        {
            return false;
        }
        m_samples = 0;
        m_events = 0;
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

    WFTRequest_GetEvent::SPtr request = m_reqBuilder->buildRequest_GET_EVT( m_events, m_svr_samp_evt.nevents - 1 );
    WFTResponse::SPtr response( new WFTResponse );

    if( !doRequest( *request, *response ) )
    {
        return false;
    }

    m_ftEvents->clear();
    if( !m_ftEvents->parseResponse( response ) )
    {
        return false;
    }

    m_events = m_svr_samp_evt.nevents;

    request.reset();
    response.reset();

    return true;
}

UINT32_T WFTRtClient::getTimeout() const
{
    return m_waitTimeout_ms;
}

void WFTRtClient::setTimeout( UINT32_T timeout )
{
    m_waitTimeout_ms = timeout;
}

bool WFTRtClient::doFlushHeaderRequest()
{
    return doFlush( FLUSH_HDR );
}

bool WFTRtClient::doFlushDataRequest()
{
    return doFlush( FLUSH_DAT );
}

bool WFTRtClient::doFlushEventsRequest()
{
    return doFlush( FLUSH_EVT );
}

bool WFTRtClient::doFlush( UINT16_T command )
{
    WFTRequest::SPtr request( new WFTRequest );
    request->getMessageDef().command = command;
    request->getMessageDef().bufsize = 0;
    request->getMessageDef().version = VERSION;

    WFTResponse::SPtr response( new WFTResponse );

    if( !doRequest( *request, *response ) )
    {
        return false;
    }

    return response->checkFlush();
}
