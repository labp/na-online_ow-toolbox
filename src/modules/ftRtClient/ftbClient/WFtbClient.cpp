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

#include <boost/foreach.hpp>

#include <buffer.h>

#include <core/common/WLogger.h>

#include "chunkReader/WFTChunkReaderChanNames.h"
#include "chunkReader/WFTChunkReaderNeuromagHdr.h"
#include "chunkReader/WFTChunkReaderNeuromagIsotrak.h"
#include "container/WFTEventIterator.h"
#include "WFtbClient.h"

const std::string WFtbClient::CLASS = "WFtbClient";

const float WFtbClient::DEFAULT_WAIT_TIMEOUT = 40;

WFtbClient::WFtbClient() :
                WLRtClient()
{
    m_samples = 0;
    m_eventCount = 0;
    m_svr_samp_evt.nsamples = 0;
    m_svr_samp_evt.nevents = 0;
    m_timeout = DEFAULT_WAIT_TIMEOUT;

    WFTChunkReader::SPtr chunkReader;
    chunkReader.reset( new WFTChunkReaderChanNames );
    m_chunkReader.insert( WFTChunkReader::MapT::value_type( chunkReader->supportedChunkType(), chunkReader ) );
    chunkReader.reset( new WFTChunkReaderNeuromagHdr );
    m_chunkReader.insert( WFTChunkReader::MapT::value_type( chunkReader->supportedChunkType(), chunkReader ) );
    chunkReader.reset( new WFTChunkReaderNeuromagIsotrak );
    m_chunkReader.insert( WFTChunkReader::MapT::value_type( chunkReader->supportedChunkType(), chunkReader ) );
}

WFtbClient::~WFtbClient()
{
    if( isStreaming() )
    {
        stop();
    }
    if( isConnected() )
    {
        disconnect();
    }
}

void WFtbClient::setConnection( WFTConnection::SPtr con )
{
    if( getStatus() == STATUS_DISCONNECTED )
    {
        m_connection = con;
    }
    else
    {
        wlog::error( CLASS ) << __func__ << ": Could not set connection!";
    }
}

void WFtbClient::setTimeout( float timeout )
{
    m_timeout = timeout;
}

bool WFtbClient::connect()
{
    if( !m_connection )
    {
        wlog::error( CLASS ) << "No connection available!";
        m_status = STATUS_DISCONNECTED;
        return false;
    }
    if( isConnected() )
    {
        wlog::info( CLASS ) << "Already connected!";
        return true;
    }

    if( m_connection->connect() )
    {
        m_status = STATUS_CONNECTED;
        wlog::info( CLASS ) << "Connection established!";
        return true;
    }
    else
    {
        m_status = STATUS_DISCONNECTED;
        wlog::error( CLASS ) << "Could not connect!";
        return false;
    }
}

void WFtbClient::disconnect()
{
    if( !m_connection )
    {
        wlog::error( CLASS ) << "No connection available!";
        m_status = STATUS_DISCONNECTED;
        return;
    }
    if( isStreaming() )
    {
        if( !stop() )
        {
            wlog::error( CLASS ) << "Could not disconnect!";
            return;
        }
    }
    m_connection->disconnect();
    m_status = STATUS_DISCONNECTED;
    wlog::info( CLASS ) << "Disconnected!";
}

bool WFtbClient::start()
{
    wlog::debug( CLASS ) << __func__ << "() called!";

    if( !isConnected() )
    {
        wlog::info( CLASS ) << "Client is not connected. Trying to connect ...";
        if( !connect() )
        {
            wlog::error( CLASS ) << "Error while connecting to the FieldTrip Buffer Server. Client is disconnect.";
            return false;
        }
    }

    if( isStreaming() )
    {
        wlog::info( CLASS ) << "Could not start streaming. Client is already streaming!";
        return true;
    }

    wlog::info( CLASS ) << "Prepare streaming.";
    if( doHeaderRequest() )
    {
        m_status = STATUS_STREAMING;
        wlog::info( CLASS ) << "Preparation for streaming finished. Header information are ready to retrieval.";
        return true;
    }
    else
    {
        wlog::error( CLASS ) << "Error while Preparation.";
        return false;
    }
}

bool WFtbClient::stop()
{
    m_status = STATUS_STOPPED;
    return true;
}

bool WFtbClient::fetchData()
{
    if( doWaitRequest( m_samples, m_eventCount ) )
    {
        doGetEventsRequest();
        return doGetDataRequest();
    }
    else
    {
        return false;
    }
}

bool WFtbClient::readEmm( WLEMMeasurement::SPtr emm )
{
    WLEMDRaw::SPtr rawData;
    if( !getRawData( &rawData ) )
    {
        return false;
    }

    bool rc = false;
    for( WFTChunkReader::MapT::iterator it = m_chunkReader.begin(); it != m_chunkReader.end(); ++it )
    {
        if( it->second->apply( emm, rawData ) )
        {
            rc |= true;
        }
    }

    if( !rc )
    {
        emm->addModality( rawData );
        wlog::debug( CLASS ) << "create raw EMM";
    }

    if( m_svr_samp_evt.nevents > m_eventCount ) // hasNewEvents
    {
        BOOST_FOREACH( WFTEvent::SPtr event, m_events ){
        wlog::debug( CLASS ) << "Fire Event: " << *event;
    }
}
    return rc;
}

bool WFtbClient::getRawData( WLEMDRaw::SPtr* const rawData )
{
    rawData->reset( new WLEMDRaw );

    if( m_data->getDataDef().bufsize == 0 )
    {
        return false;
    }

    int chans = m_data->getDataDef().nchans;
    int samps = m_data->getDataDef().nsamples;
    ScalarT *dataSrc;
    SimpleStorage floatStore;

    // convert data to the used floating point number format
    if( m_data->needDataToConvert< ScalarT >() )
    {
        floatStore.resize( sizeof(ScalarT) * samps * chans );
        dataSrc = ( ScalarT * )floatStore.data();

        m_data->convertData< ScalarT >( dataSrc, m_data->getData(), samps, chans, m_data->getDataDef().data_type );
    }
    else // data arrived in the right format
    {
        dataSrc = ( ScalarT * )m_data->getData();
    }

    WLEMData::DataSPtr dataPtr( new WLEMData::DataT( chans, samps ) ); // create data matrix
    WLEMData::DataT& data = *dataPtr;

    ( *rawData )->setSampFreq( m_header->getHeaderDef().fsample );

    // insert value into the matrix
    for( int i = 0; i < samps; ++i ) // iterate all samples
    {
        // point to the next samples start position
        const ScalarT *sampleSrc = dataSrc + i * chans;
        WLEMData::SampleT sample( chans, 1 ); // create new sample for the data matrix

        for( int j = 0; j < chans; ++j )
        {
            ScalarT x = sampleSrc[j];

#ifdef LABP_FLOAT_COMPUTATION
            if( isnanf( x ) || isinff( x ) )
            {
                return false;
            }
#else
            if( isnan( x ) || isinf( x ) )
            {
                return false;
            }
#endif  // LABP_FLOAT_COMPUTATION

            sample( j ) = x; // copy the samples values into the vector
        }

        data.col( i ) = sample; // add sample-vector to the matrix
    }

    ( *rawData )->setData( dataPtr );

    return true;
}

bool WFtbClient::doRequest( WFTResponse* const response, const WFTRequest& request )
{
    // Lock the client for thread-safe requests.
    boost::unique_lock< boost::shared_mutex > unqLock( m_requestLock );

    if( !isConnected() )
    {
        wlog::error( CLASS ) << "The client is not connected.";

        return false;
    }

    if( clientrequest( m_connection->getSocket(), request.out(), response->in() ) < 0 )
    {
        wlog::error( CLASS ) << "Error in communication - check buffer server.";
        return false;
    }

    unqLock.unlock();

    return response->isValid();
}

bool WFtbClient::doHeaderRequest()
{
    WFTResponse response;
    WFTRequest request;
    request.prepGetHeader();
    m_header.reset( new WFTHeader );

    if( !doRequest( &response, request) )
    {
        return false;
    }

    if( !m_header->parseResponse( response ) )
    {
        wlog::error( CLASS ) << "Error while parsing server response.";
        return false;
    }

    // check for new samples & events
    m_svr_samp_evt.nsamples = m_header->getHeaderDef().nsamples;
    m_svr_samp_evt.nevents = m_header->getHeaderDef().nevents;

    WLList< WFTChunk::SPtr >::ConstSPtr chunks = m_header->getChunks();
    for( WLList< WFTChunk::SPtr >::const_iterator it = chunks->begin(); it != chunks->end(); ++it )
    {
        const wftb::chunk_type_t chunk_type = ( *it )->getChunkType();
        if( m_chunkReader.count( chunk_type ) > 0 )
        {
            m_chunkReader[chunk_type]->read( *it );
        }
    }

    return true;
}

bool WFtbClient::doWaitRequest( wftb::nsamples_t samples, wftb::nevents_t events )
{
    WFTResponse response;
    WFTRequest request;
    request.prepWaitData( samples, events, m_timeout );

    if( !doRequest( &response, request ) )
    {
        wlog::error( CLASS ) << "Error while doing Wait-Request.";

        return false;
    }

    if( !response.checkWait( m_svr_samp_evt.nsamples, m_svr_samp_evt.nevents ) )
    {
        wlog::error( CLASS ) << "Error while checking Wait-Request response.";

        wlog::error( CLASS ) << response;
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
        m_eventCount = 0;
    }

    return true;
}

bool WFtbClient::doGetDataRequest()
{
    if( m_header == 0 )
    {
        return false;
    }

    if( !( m_svr_samp_evt.nsamples > m_samples ) ) // !(hasNewSamples)
    {
        return false;
    }

    // calculate the last samples index depending on the sampling frequency and the number of store sample on the server.
    wftb::nsamples_t endSample =
                    m_svr_samp_evt.nsamples - m_samples >= m_header->getHeaderDef().fsample ? m_samples
                                    + m_header->getHeaderDef().fsample - 1 :
                                    m_svr_samp_evt.nsamples - 1;
    //UINT32_T endSample = m_svr_samp_evt.nsamples - 1;
    WFTRequest request;
    request.prepGetData( m_samples, endSample );
    WFTResponse response;

    if( !doRequest( &response, request ) )
    {
        return false;
    }

    m_data.reset( new WFTData );

    if( !m_data->parseResponse( response ) )
    {
        return false;
    }

    //m_samples = m_svr_samp_evt.nsamples; // update number of read samples.
    m_samples = endSample + 1; // update number of read samples.

    return true;
}

bool WFtbClient::doGetEventsRequest()
{
    if( m_header == 0 )
    {
        return false;
    }

    if( !( m_svr_samp_evt.nevents > m_eventCount ) ) // !(hasNewEvents)
    {
        return false;
    }

    WFTRequest request;
    request.prepGetEvents( m_eventCount, m_svr_samp_evt.nevents - 1 );
    WFTResponse response;

    if( !doRequest( &response, request ) )
    {
        return false;
    }

    m_events.clear();
    SimpleStorage storage;
    if( response.checkGetEvents( &storage ) < 0 )
    {
        return false;
    }

    WFTEventIterator it( &storage, response.m_response->def->bufsize);
    while( it.hasNext() )
    {
        WFTEvent::SPtr evt = it.getNext();
        if( evt != NULL )
        {
            m_events.push_back( evt );
        }
    }
    m_eventCount = m_svr_samp_evt.nevents;

    return true;
}
