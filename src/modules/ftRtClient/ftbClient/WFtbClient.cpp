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
#include "util/WFTEventIterator.h"
#include "WFtbClient.h"

const std::string WFtbClient::CLASS = "WFtbClient";

WFtbClient::WFtbClient() :
                WLRtClient()
{
    reset();
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
    reset();
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
        WLList< WFTChunk::SPtr >::ConstSPtr chunks = m_header->getChunks();
        for( WLList< WFTChunk::SPtr >::const_iterator it = chunks->begin(); it != chunks->end(); ++it )
        {
            const wftb::chunk_type_t chunk_type = ( *it )->getChunkType();
            if( m_chunkReader.count( chunk_type ) > 0 )
            {
                m_chunkReader[chunk_type]->read( *it );
            }
        }

        m_status = STATUS_STREAMING;
        m_timeout = 1000.0 * ( 2.0 * ( m_blockSize / m_header->getHeaderDef().fsample ) ); // wait for max 2 blocks in ms.

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
    if( doWaitRequest( m_idxSamples + getBlockSize() - 1, m_idxEvents, m_timeout ) )
    {
        doGetEventsRequest( m_idxEvents, m_svr_samp_evt.nevents - 1 );
        return doGetDataRequest( m_idxSamples, m_idxSamples + m_blockSize - 1 );
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

    if( m_svr_samp_evt.nevents > m_idxEvents ) // hasNewEvents
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

    if( m_data->getDataSize() == 0 )
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

    if( !doRequest( &response, request ) )
    {
        return false;
    }

    if( !m_header->deserialize( response ) )
    {
        wlog::error( CLASS ) << "Error while parsing server response.";
        return false;
    }

    // check for new samples & events
    m_svr_samp_evt.nsamples = m_header->getHeaderDef().nsamples;
    m_svr_samp_evt.nevents = m_header->getHeaderDef().nevents;

    return true;
}

bool WFtbClient::doWaitRequest( wftb::nsamples_t samples, wftb::nevents_t events, wftb::time_t timeout )
{
    WFTResponse response;
    WFTRequest request;
    request.prepWaitData( samples, events, timeout );

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
        wlog::debug( CLASS ) << "No new data available!";
        return false;
    }

    return true;
}

bool WFtbClient::doGetDataRequest( wftb::isample_t begin, wftb::isample_t end )
{
    if( m_header == 0 )
    {
        return false;
    }

    if( !( end < m_svr_samp_evt.nsamples ) ) // has new samples?
    {
        return false;
    }

    WFTRequest request;
    request.prepGetData( begin, end );
    WFTResponse response;

    if( !doRequest( &response, request ) )
    {
        return false;
    }

    m_data.reset( new WFTData );

    if( !m_data->deserialize( response ) )
    {
        return false;
    }

    m_idxSamples = end + 1; // update number of read samples.

    return true;
}

bool WFtbClient::doGetEventsRequest( wftb::ievent_t begin, wftb::ievent_t end )
{
    if( m_header == 0 )
    {
        return false;
    }

    if( !( m_svr_samp_evt.nevents > begin ) ) // !(hasNewEvents)
    {
        return false;
    }

    WFTRequest request;
    request.prepGetEvents( begin, end );
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

    WFTEventIterator it( &storage, response.m_response->def->bufsize );
    while( it.hasNext() )
    {
        WFTEvent::SPtr evt = it.getNext();
        if( evt != NULL )
        {
            m_events.push_back( evt );
        }
    }
    m_idxEvents = end + 1;

    return true;
}

void WFtbClient::reset()
{
    wlog::debug( CLASS ) << __func__ << "() called!";
    m_applyScaling = false;
    m_idxSamples = 0;
    m_idxEvents = 0;
    m_svr_samp_evt.nsamples = 0;
    m_svr_samp_evt.nevents = 0;
    m_timeout = 500.0;

    m_chunkReader.clear();
    WFTChunkReader::SPtr chunkReader;
    chunkReader.reset( new WFTChunkReaderChanNames );
    m_chunkReader.insert( WFTChunkReader::MapT::value_type( chunkReader->supportedChunkType(), chunkReader ) );
    WFTChunkReaderNeuromagHdr::SPtr chunkNeuromag( new WFTChunkReaderNeuromagHdr );
    chunkNeuromag->setApplyScaling( m_applyScaling );
    m_chunkReader.insert( WFTChunkReader::MapT::value_type( chunkNeuromag->supportedChunkType(), chunkNeuromag ) );
    chunkReader.reset( new WFTChunkReaderNeuromagIsotrak );
    m_chunkReader.insert( WFTChunkReader::MapT::value_type( chunkReader->supportedChunkType(), chunkReader ) );
}

void WFtbClient::setApplyScaling( bool apply )
{
    if( m_applyScaling == apply )
    {
        return;
    }
    m_applyScaling = apply;
    if( m_chunkReader.count( wftb::ChunkType::NEUROMAG_HEADER ) > 0 )
    {
        boost::dynamic_pointer_cast< WFTChunkReaderNeuromagHdr >( m_chunkReader[wftb::ChunkType::NEUROMAG_HEADER] )->setApplyScaling(
                        apply );
    }
}
