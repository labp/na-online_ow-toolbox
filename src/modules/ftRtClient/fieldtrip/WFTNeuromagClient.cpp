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

#include <boost/foreach.hpp>

#include <core/common/WLogger.h>

#include "core/data/WLDataTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"

#include "dataTypes/WFTChunk.h"
#include "WFTNeuromagClient.h"

const std::string WFTNeuromagClient::CLASS = "WFTClientStreaming";

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

    //wlog::debug( CLASS ) << "Test for convert (used Type in Openwalnut: " << typeid(ScalarT).name() << ")";

    // convert data to the used floating point number format
    if( m_ftData->needDataToConvert< ScalarT >() )
    {
        //wlog::debug( CLASS ) << "Data has to convert in: " << typeid(ScalarT).name();
        //wlog::debug( CLASS ) << "sizeof(ScalarT) = " << sizeof(ScalarT);

        floatStore.resize( sizeof(ScalarT) * samps * chans );
        dataSrc = ( ScalarT * )floatStore.data();

        m_ftData->convertData< ScalarT >( dataSrc, m_ftData->getData(), samps, chans, m_ftData->getDataDef().data_type );
    }
    else // data arrived in the right format
    {
        //wlog::debug( CLASS ) << "Data does not need to convert, because: " << typeid(ScalarT).name();

        dataSrc = ( ScalarT * )m_ftData->getData();
    }

    WLEMData::SPtr modality( new WLEMDEEG );

    WLEMData::DataSPtr dataPtr( new WLEMData::DataT( chans, samps ) ); // create data matrix
    WLEMData::DataT& data = *dataPtr;

    modality->setSampFreq(m_ftHeader->getHeaderDef().fsample);

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
                //wlog::debug( CLASS ) << "Float NoN";

                return false;
            }
#else
            if( isnan( x ) || isinf( x ) )
            {
                //wlog::debug( CLASS ) << "Double NoN";

                return false;
            }
#endif  // LABP_FLOAT_COMPUTATION

            sample( j ) = x; // copy the samples values into the vector
        }

        data.col( i ) = sample; // add sample-vector to the matrix
    }

    modality->setData( dataPtr );
    emm.addModality( modality );

    //wlog::debug( CLASS ) << "EMM creation successful";

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
        wlog::debug( CLASS ) << "Chunk [" << WLEFTChunkType::name( chunk->getType() ) << "]:";

        switch( chunk->getType() )
        {
            case WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES:
                wlog::debug( CLASS ) << chunk->getDataString();
                break;
            case WLEFTChunkType::FT_CHUNK_CHANNEL_FLAGS:
                wlog::debug( CLASS ) << chunk->getDataString();
                break;
            case WLEFTChunkType::FT_CHUNK_RESOLUTIONS:
                wlog::debug( CLASS ) << chunk->getDataString();
                break;
            case WLEFTChunkType::FT_CHUNK_ASCII_KEYVAL:
                wlog::debug( CLASS ) << chunk->getDataString();
                break;
            case WLEFTChunkType::FT_CHUNK_NIFTI1:
                wlog::debug( CLASS ) << "NIFTI-1 file.";
                break;
            case WLEFTChunkType::FT_CHUNK_SIEMENS_AP:
                wlog::debug( CLASS ) << chunk->getDataString();
                break;
            case WLEFTChunkType::FT_CHUNK_CTF_RES4:
                wlog::debug( CLASS ) << "CTF .res4 file.";
                break;
            case WLEFTChunkType::FT_CHUNK_NEUROMAG_ISOTRAK:
                wlog::debug( CLASS ) << "Neuromag Isotrak .fif file.";
                break;
            case WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER:
                wlog::debug( CLASS ) << "Neuromag header .fif file.";
                break;
            case WLEFTChunkType::FT_CHUNK_NEUROMAG_HPIRESULT:
                wlog::debug( CLASS ) << "Neuromag HPI result .fif file.";
                break;
            default:
                break;
        }
    }

}

bool WFTNeuromagClient::prepareStreaming()
{
    return doHeaderRequest();
}
