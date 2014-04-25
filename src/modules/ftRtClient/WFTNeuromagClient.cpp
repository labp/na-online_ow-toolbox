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
#include <fstream>
#include <map>
#include <sstream>

#include <boost/foreach.hpp>

#include <Eigen/src/Core/util/Constants.h>

#include <core/common/WLogger.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDECG.h"
#include "core/data/enum/WLEModality.h"
#include "core/data/WLDataTypes.h"
#include "core/dataFormat/fiff/WLFiffChType.h"

#include "fieldtrip/dataTypes/chunks/WFTChunk.h"
#include "fieldtrip/dataTypes/chunks/WFTChunkFactory.h"
#include "fieldtrip/dataTypes/chunks/WFTChunkNeuromagHdr.h"
#include "fieldtrip/dataTypes/chunks/WFTChunkNeuromagIsotrak.h"
#include "WFTNeuromagClient.h"

const std::string WFTNeuromagClient::CLASS = "WFTNeuromagClient";

WFTNeuromagClient::WFTNeuromagClient() :
                m_streaming( false )
{

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
    WLEMDRaw::SPtr rawData;

    if( !getRawData( rawData ) )
    {
        return false;
    }

    // if there is no Neuromag header, return raw data.
    if( !m_header->hasChunk( WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER ) )
    {
        emm.addModality( rawData );

        wlog::debug( CLASS ) << "create raw EMM";

        return true;
    }

    return createDetailedEMM( emm, rawData );
}

void WFTNeuromagClient::printChunks()
{
    if( m_header == 0 )
    {
        return;
    }

    if( !m_header->hasChunks() )
    {
        return;
    }

    // todo(maschke): print chunks
    BOOST_FOREACH(WFTAChunk::SPtr chunk, *m_header->getChunks())
    {
        std::string str;
        str += "Chunk [Type=" + WLEFTChunkType::name( chunk->getType() ) + "]";

        wlog::debug( CLASS ) << str;
    }
}

bool WFTNeuromagClient::getRawData( WLEMDRaw::SPtr& modality )
{
    modality.reset( new WLEMDRaw );

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

    modality->setSampFreq( m_header->getHeaderDef().fsample );

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

    return true;
}

bool WFTNeuromagClient::createDetailedEMM( WLEMMeasurement& emm, WLEMDRaw::SPtr rawData )
{
    wlog::debug( CLASS ) << "create full detailed EMM";

    WFTChunkNeuromagHdr::SPtr neuromagHdr = m_header->getChunks( WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER )->at( 0 )->getAs<
                    WFTChunkNeuromagHdr >();

    //
    //  transfer data for all modalities and add channel names if exist.
    //
    for( std::map< WLEModality::Enum, WLEMDRaw::ChanPicksT >::iterator it = neuromagHdr->getModalityPicks()->begin();
                    it != neuromagHdr->getModalityPicks()->end(); ++it )
    {
        WLEMData::SPtr emd;

        switch( it->first )
        {
            case WLEModality::EEG:
                emd.reset( new WLEMDEEG() );
                break;
            case WLEModality::MEG:
                emd.reset( new WLEMDMEG() );
                break;
            case WLEModality::EOG:
                emd.reset( new WLEMDEOG() );
                break;
            case WLEModality::ECG:
                emd.reset( new WLEMDECG() );
                break;
            default:
                continue;
        }

        emd->setData( rawData->getData( it->second, true ) );

        WLArrayList< std::string >::SPtr channelNames = neuromagHdr->getChannelNames( it->first );

        if( channelNames != 0 && channelNames->size() > 0 )
        {
            emd->setChanNames( channelNames );
        }

        emm.addModality( emd );
    }

    //
    //  Add event / stimulus channels to the EMM
    //
    if( neuromagHdr->getStimulusPicks()->cols() > 0 )
    {
        emm.setEventChannels( readEvents( ( Eigen::MatrixXf& )rawData->getData(), *neuromagHdr->getStimulusPicks() ) );
    }

    //
    //  Add digitalization points.
    //
    if( m_header->hasChunk( WLEFTChunkType::FT_CHUNK_NEUROMAG_ISOTRAK ) )
    {
        emm.setDigPoints(
                        m_header->getChunks( WLEFTChunkType::FT_CHUNK_NEUROMAG_ISOTRAK )->at( 0 )->getAs< WFTChunkNeuromagIsotrak >()->getData() );
    }

    return true;
}

bool WFTNeuromagClient::prepareStreaming()
{
    return doHeaderRequest();
}
