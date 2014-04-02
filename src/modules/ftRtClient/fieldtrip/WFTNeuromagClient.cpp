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

#include <cmath>
#include <sstream>

#include <boost/foreach.hpp>

#include <core/common/WLogger.h>

#include "core/data/WLDataTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"

#include "dataTypes/WFTChunk.h"
#include "WFTNeuromagClient.h"

const std::string WFTNeuromagClient::CLASS = "WFTNeuromagClient";

WFTNeuromagClient::WFTNeuromagClient()
{
    m_streaming = false;
}

bool WFTNeuromagClient::isStreaming() const
{
    return m_streaming;
}

bool WFTNeuromagClient::start()
{
    wlog::debug( CLASS ) << "start() called!";

    resetClient(); // reset the base client class.

    if( isStreaming() )
    {
        wlog::info( CLASS ) << "Could not start streaming. Client is already streaming!";
        return true;
    }

    if( !isConnected() )
    {
        wlog::info( CLASS ) << "Client is not connected. Client is trying to connect.";

        if( !this->connect() )
        {
            wlog::error( CLASS ) << "Error while connecting to the FieldTrip Buffer Server. Client is disconnect.";
            return false;
        }
    }

    wlog::info( CLASS ) << "Prepare streaming.";
    if( prepareStreaming() )
    {
        wlog::info( CLASS ) << "Preparation for streaming finished. Header information are ready to retrieval.";
    }
    else
    {
        wlog::error( CLASS ) << "Error while Preparation.";
        return false;
    }

    return m_streaming = true;
}

void WFTNeuromagClient::stop()
{
    m_streaming = false;
}

bool WFTNeuromagClient::createEMM( WLEMMeasurement& emm )
{
    if( m_ftData->getDataDef().bufsize == 0 )
    {
        return false;
    }

    int chans = m_ftData->getDataDef().nchans;
    int samps = m_ftData->getDataDef().nsamples;
    ScalarT *dataSrc;
    SimpleStorage floatStore;

    // convert data to the used floating point number format
    if( m_ftData->needDataToConvert< ScalarT >() )
    {
        floatStore.resize( sizeof(ScalarT) * samps * chans );
        dataSrc = ( ScalarT * )floatStore.data();

        m_ftData->convertData< ScalarT >( dataSrc, m_ftData->getData(), samps, chans, m_ftData->getDataDef().data_type );
    }
    else // data arrived in the right format
    {
        dataSrc = ( ScalarT * )m_ftData->getData();
    }

    WLEMData::SPtr modality( new WLEMDEEG );

    WLEMData::DataSPtr dataPtr( new WLEMData::DataT( chans, samps ) ); // create data matrix
    WLEMData::DataT& data = *dataPtr;

    modality->setSampFreq( m_ftHeader->getHeaderDef().fsample );

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

    modality->setData( dataPtr );
    emm.addModality( modality );

    return true;
}

void WFTNeuromagClient::printChunks()
{
    if( m_ftHeader == 0 )
    {
        return;
    }

    if( !m_ftHeader->hasChunks() )
    {
        return;
    }

    BOOST_FOREACH(WFTChunk::SPtr chunk, *m_ftHeader->getChunks())
    {
        std::string str;
        str += "Chunk [Type=" + WLEFTChunkType::name( chunk->getType() ) + "]: " + chunk->getDataString();

        wlog::debug( CLASS ) << str;
    }
}

bool WFTNeuromagClient::prepareStreaming()
{
    return doHeaderRequest();
}
