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

#include <core/common/WLogger.h>

#include "core/data/WLDataTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"

#include "fieldtrip/dataTypes/chunks/WFTChunk.h"
#include "fieldtrip/processing/WFTChunkProcessor.h"
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
    if( m_header->getMeasurementInfo() == 0 )
    {
        emm.addModality( rawData );

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

    BOOST_FOREACH(WFTChunk::SPtr chunk, *m_header->getChunks())
    {
        std::string str;
        str += "Chunk [Type=" + WLEFTChunkType::name( chunk->getType() ) + "]: " + chunk->getDataString();

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

    // todo: Die Aufteilung in Modalitäten/ Channel Typen vornehmen -> am besten in map<Modality, WLEMDRaw::ChanPicksT>  = Eigen::VectorXi
    std::map< WLEModality::Enum, WLEMDRaw::ChanPicksT > modalityPicks;

    for( int i = 0; i < m_header->getMeasurementInfo()->chs.size(); ++i )
    {
        FIFFLIB::FiffChInfo info = m_header->getMeasurementInfo()->chs.at( i );

        WLEModality::Enum modalityType = WLEModality::fromFiffType( info.kind );

        if( modalityPicks.count( modalityType ) == 0 )
        {
            modalityPicks.insert(
                            std::map< WLEModality::Enum, WLEMDRaw::ChanPicksT >::value_type( modalityType,
                                            WLEMDRaw::ChanPicksT() ) );
        }

        WLEMDRaw::ChanPicksT& vector = modalityPicks.at( modalityType );
        vector.resize( vector.rows(), vector.cols() + 1 );
        vector( vector.cols() - 1 ) = i;

    }

    //wlog::debug( CLASS ) << modalityPicks.at( WLEModality::MEG );

    // todo: über die Picks die Daten modalitätenweise holen.

    // todo: analog die Channel Names.

    return true;
}

bool WFTNeuromagClient::prepareStreaming()
{
    if( !doHeaderRequest() )
    {
        return false;
    }

    WFTChunkProcessor::SPtr processor( new WFTChunkProcessor );

    if( m_header->hasChunk( WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES ) )
    {
        WLArrayList< std::string >::SPtr list( new WLArrayList< std::string > );

        if( processor->channelNamesChunk( m_header->getChunks( WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES )->at( 0 ), list ) )
        {
            // todo(maschke): include channel flags chunk during FieldTrip channel names processing.
        }

    }

    // receive Neuromag files.
    if( m_header->hasChunk( WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER ) )
    {
        if( processor->processNeuromagHeader( m_header->getChunks( WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER )->at( 0 ) ) )
        {
            m_header->setMeasurementInfo( processor->getMeasurementInfo() );
        }

    }

    return true;
}
