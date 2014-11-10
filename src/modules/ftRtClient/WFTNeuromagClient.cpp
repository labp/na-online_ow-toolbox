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
#include "WFTNeuromagClient.h"
#include "ftbClient/dataTypes/chunks/WFTChunkNeuromagHdr.h"
#include "ftbClient/dataTypes/chunks/WFTChunkNeuromagIsotrak.h"

const std::string WFTNeuromagClient::CLASS = "WFTNeuromagClient";

WFTNeuromagClient::WFTNeuromagClient() :
                m_streaming( false ), m_applyScaling( false )
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

bool WFTNeuromagClient::createEMM( WLEMMeasurement::SPtr emm )
{
    WLEMDRaw::SPtr rawData;

    if( !getRawData( &rawData ) )
    {
        return false;
    }

    // if there is no Neuromag header, return raw data.
    if( !m_header->hasChunk( wftb::ChunkType::NEUROMAG_HEADER ) )
    {
        emm->addModality( rawData );

        wlog::debug( CLASS ) << "create raw EMM";

        return true;
    }

    return createDetailedEMM( emm, rawData );
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
        floatStore.resize( sizeof( ScalarT ) * samps * chans );
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

bool WFTNeuromagClient::createDetailedEMM( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr rawData )
{
    // get Neuromag header
    WFTChunkNeuromagHdr::SPtr neuromagHdr = m_header->getChunks( wftb::ChunkType::NEUROMAG_HEADER )->at( 0 )->getAs<
                    WFTChunkNeuromagHdr >();

    // get Neuromag Isotrak
    WFTChunkNeuromagIsotrak::SPtr isotrak;
    if( m_header->hasChunk( wftb::ChunkType::NEUROMAG_ISOTRAK ) )
    {
        isotrak = m_header->getChunks( wftb::ChunkType::NEUROMAG_ISOTRAK )->at( 0 )->getAs< WFTChunkNeuromagIsotrak >();
    }

    //
    //  Add digitalization points.
    //
    if( isotrak )
    {
        emm->setDigPoints( isotrak->getDigPoints() );
    }

    //
    //  transfer data for all modalities and add channel names if exist.
    //
    std::map< WLEModality::Enum, WLEMDRaw::ChanPicksT >::const_iterator it;
    for( it = neuromagHdr->getModalityPicks()->begin(); it != neuromagHdr->getModalityPicks()->end(); ++it )
    {
        // skip unknown modalities: in case of Neuromag - the "Misc" channel.
        if( it->first == WLEModality::UNKNOWN )
        {
            continue;
        }

        //
        // Create the data container.
        //
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
            case WLEModality::UNKNOWN:
                emd.reset( new WLEMDRaw() );
                break;
            default:
                continue;
        }

        emd->setData( rawData->getData( it->second, true ) );
        emd->setSampFreq( rawData->getSampFreq() );

        //
        // apply scaling
        //
        if( m_applyScaling )
        {
            for( Eigen::RowVectorXi::Index row = 0; row < it->second.size(); ++row )
            {
                emd->getData().row( row ) *= neuromagHdr->getScaleFactors()->at( ( int )it->second[row] );
            }
        }

        //
        // Channel names.
        //
        WLArrayList< std::string >::SPtr channelNames = neuromagHdr->getChannelNames( it->first );
        if( channelNames )
        {
            if( channelNames->size() > 0 )
            {
                emd->setChanNames( channelNames );
            }
        }

        //
        // Channel positions.
        //
        if( emd->getModalityType() == WLEModality::EEG && isotrak ) // EEG
        {
            WLEMDEEG::SPtr eeg = emd->getAs< WLEMDEEG >();
            if( !isotrak->getEEGChanPos()->empty() )
            {
                eeg->setChannelPositions3d( isotrak->getEEGChanPos() );

                if( !isotrak->getEEGFaces()->empty() )
                {
                    eeg->setFaces( isotrak->getEEGFaces() );
                }
            }
        }

        if( emd->getModalityType() == WLEModality::MEG ) // MEG
        {
            WLEMDMEG::SPtr meg = emd->getAs< WLEMDMEG >();
            if( neuromagHdr->hasChannelPositionsMEG() )
            {
                meg->setChannelPositions3d( neuromagHdr->getChannelPositionsMEG() );
            }
            meg->setEx( neuromagHdr->getChannelExMEG() );
            meg->setEy( neuromagHdr->getChannelEyMEG() );
            meg->setEz( neuromagHdr->getChannelEzMEG() );
        }

        emm->addModality( emd ); // add modality to measurement.
    }

    //
    //  Add event / stimulus channels to the EMM
    //
    if( neuromagHdr->getStimulusPicks()->cols() > 0 )
    {
        emm->setEventChannels( readEventChannels( ( Eigen::MatrixXf& )rawData->getData(), *neuromagHdr->getStimulusPicks() ) );
    }

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
