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

#include <cmath>
#include <map>
#include <string>

#include <Eigen/Core>

#include <SimpleStorage.h>

#include <core/common/WLogger.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDECG.h"
#include "core/data/enum/WLEModality.h"
#include "core/data/WLDataTypes.h"

#include "modules/ftRtClient/ftb/WFtbChunk.h"
#include "modules/ftRtClient/ftbClient/chunkReader/WFTChunkReaderNeuromagHdr.h"
#include "modules/ftRtClient/ftbClient/chunkReader/WFTChunkReaderNeuromagIsotrak.h"
#include "WFTNeuromagClient.h"

const std::string WFTNeuromagClient::CLASS = "WFTNeuromagClient";

WFTNeuromagClient::WFTNeuromagClient() :
                m_streaming( false ), m_applyScaling( false )
{
    WFTChunkReader::SPtr chunkReader;
    chunkReader.reset( new WFTChunkReaderNeuromagHdr );
    m_chunkReader.insert( WFTChunkReader::MapT::value_type( chunkReader->supportedChunkType(), chunkReader ) );
    chunkReader.reset( new WFTChunkReaderNeuromagIsotrak );
    m_chunkReader.insert( WFTChunkReader::MapT::value_type( chunkReader->supportedChunkType(), chunkReader ) );
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

bool WFTNeuromagClient::createEMM( WLEMMeasurement::SPtr emm )
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

    return rc;
}

bool WFTNeuromagClient::getRawData( WLEMDRaw::SPtr* const rawData )
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

bool WFTNeuromagClient::isScalingApplied() const
{
    return m_applyScaling;
}

void WFTNeuromagClient::setScaling( bool applyScaling )
{
    m_applyScaling = applyScaling;
}

bool WFTNeuromagClient::prepareStreaming()
{
    return doHeaderRequest();
}
