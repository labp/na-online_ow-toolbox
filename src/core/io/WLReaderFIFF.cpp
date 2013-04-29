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
#include <set>

#include <boost/shared_ptr.hpp>

#include <libfiffio/common/LFDigitisationPoint.h>
#include <libfiffio/common/LFInterface.h>
#include <libfiffio/common/LFIsotrak.h>
#include <libfiffio/common/LFReturnCodes.h>
#include <libfiffio/common/LFUnits.h>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/common/WLogger.h>
#include <core/common/WAssert.h>

#include "core/dataHandler/WDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMSubject.h"
#include "core/dataHandler/WDataSetEMMEMD.h"
#include "core/dataHandler/WDataSetEMMEEG.h"
#include "core/dataHandler/WDataSetEMMMEG.h"
#include "core/dataHandler/WDataSetEMMEOG.h"
#include "core/dataHandler/WDataSetEMMECG.h"
#include "core/dataHandler/WDataSetEMMEnumTypes.h"

#include "WLReaderFIFF.h"

using namespace LaBP;

const std::string WLReaderFIFF::CLASS = "WLReaderFIFF";

WLReaderFIFF::WLReaderFIFF( std::string fname ) :
                WLReader( fname )
{
    wlog::debug( CLASS ) << "file: " << fname;
}

WLReaderFIFF::ReturnCode::Enum WLReaderFIFF::Read( LaBP::WDataSetEMM::SPtr out )
{
    LFData data;
    returncode_t ret = LFInterface::fiffRead( data, m_fname.data() );
    if( ret != rc_normal )
        return getReturnCode( ret );

    // Set subject information
    LaBP::WDataSetEMMSubject::SPtr subject_out( new LaBP::WDataSetEMMSubject() );
    ReturnCode::Enum rc = Read( subject_out );
    if( rc != ReturnCode::SUCCESS )
        return rc;
    out->setSubject( subject_out );

    // Create temporary EMMEMD
    LaBP::WDataSetEMMEEG::SPtr dummy( new LaBP::WDataSetEMMEEG() );
    LFMeasurementInfo& measinfo_in = data.GetLFMeasurement().GetLFMeasurementInfo();
    int32_t nChannels = measinfo_in.GetNumberOfChannels();
    wlog::debug( CLASS ) << "Channels: " << nChannels;
    dummy->setSampFreq( measinfo_in.GetSamplingFrequency() );
    dummy->setAnalogHighPass( measinfo_in.GetHighpass() );
    dummy->setAnalogLowPass( measinfo_in.GetLowpass() );
    dummy->setLineFreq( measinfo_in.GetLineFreq() );

    // Read Isotrak
    LFIsotrak& isotrak = measinfo_in.GetLFIsotrak();
    LFArrayPtr< LFDigitisationPoint >& digPoints = isotrak.GetLFDigitisationPoint();
    boost::shared_ptr< std::vector< WVector3f > > itPos( new std::vector< WVector3f >() );
    for( LFArrayPtr< LFDigitisationPoint >::size_type i = 0; i < digPoints.size(); ++i )
    {
        itPos->push_back( WVector3f( digPoints[i]->GetRr()[0], digPoints[i]->GetRr()[1], digPoints[i]->GetRr()[2] ) );
    }
    subject_out->setIsotrak( itPos );
    wlog::debug( CLASS ) << "Isotrak size: " << itPos->size();

    // Read raw data
    LFRawData& rawdata_in = data.GetLFMeasurement().GetLFRawData();
    LFArrayPtr< LFDataBuffer >& rawdatabuffers_in = rawdata_in.GetLFDataBuffer();
    size_t nBuffers_in = rawdatabuffers_in.size();
    size_t nBuffers_out = 0;
    for( size_t i = 0; i < nBuffers_in; i++ )
        nBuffers_out += rawdatabuffers_in[i]->GetSize();
    nBuffers_out /= nChannels;
    LaBP::WDataSetEMMEMD::DataT rawdatabuffers_out( nChannels );
    for( int32_t i = 0; i < nChannels; i++ )
        rawdatabuffers_out[i].resize( nBuffers_out );
    int32_t current_channel = 0, current_buffer_out = 0;
    LFArrayPtr< LFChannelInfo >& channelInfos = measinfo_in.GetLFChannelInfo();
    double scaleFactor;
    for( size_t i = 0; i < nBuffers_in; i++ )
    {
        LFDataBuffer* pBuf = rawdatabuffers_in[i];
        size_t nValues = pBuf->GetSize();
        for( size_t j = 0; j < nValues; j++ )
        {
            scaleFactor = channelInfos[current_channel]->GetRange() * channelInfos[current_channel]->GetCal();
            switch( pBuf->GetDataType() )
            {
                case LFDataBuffer::dt_int16:
                    rawdatabuffers_out[current_channel][current_buffer_out] = pBuf->GetBufferInt16()->at( j ) * scaleFactor;
                    break;
                case LFDataBuffer::dt_int32:
                    rawdatabuffers_out[current_channel][current_buffer_out] = pBuf->GetBufferInt32()->at( j ) * scaleFactor;
                    break;
                case LFDataBuffer::dt_float:
                    rawdatabuffers_out[current_channel][current_buffer_out] = pBuf->GetBufferFloat()->at( j ) * scaleFactor;
                    break;
                default:
                    // LFDataBuffer::dt_unknown
                    break;
            }
            current_channel++;
            if( current_channel >= nChannels )
            {
                current_buffer_out++;
                current_channel = 0;
            }
        }
    }

    boost::shared_ptr< LaBP::WDataSetEMMEMD::DataT > rawdatabuffers_out_ptr(
                    new LaBP::WDataSetEMMEMD::DataT( rawdatabuffers_out ) );
    dummy->setData( rawdatabuffers_out_ptr );

    // Collect available modalities and coils
    std::set< int32_t > modalities;
    for( int32_t chan = 0; chan < nChannels; ++chan )
    {
        modalities.insert( measinfo_in.GetLFChannelInfo()[chan]->GetKind() );
    }

    // Create modality objects //
    int32_t mod;
    while( !modalities.empty() )
    {
        mod = *modalities.begin();
        modalities.erase( mod );
        LaBP::WDataSetEMMEMD::SPtr emd;
        // See FIFF PDF B.3 Channel Types
        switch( mod )
        {
            case 1: // MEG channel
                wlog::debug( CLASS ) << "Creating MEG modality ...";
                emd.reset( new LaBP::WDataSetEMMMEG() );
                break;
            case 2: // EEG channel
                wlog::debug( CLASS ) << "Creating EEG modality ...";
                emd.reset( new LaBP::WDataSetEMMEEG() );
                break;
//            case 3: // Stimulus channel
            case 202: // EOG channel
                wlog::debug( CLASS ) << "Creating EOG modality ...";
                emd.reset( new LaBP::WDataSetEMMEOG() );
                break;
            case 402: // ECG channel
                wlog::debug( CLASS ) << "Creating ECG modality ...";
                emd.reset( new LaBP::WDataSetEMMECG() );
                break;
            default:
                wlog::debug( CLASS ) << "Skip modality type: " << mod;
                continue;
        }

        // Collect data: measurement, positions, base vectors
        boost::shared_ptr< LaBP::WDataSetEMMEMD::DataT > data( new LaBP::WDataSetEMMEMD::DataT() );
        boost::shared_ptr< std::vector< WPosition > > positions( new std::vector< WPosition >() );
        float* pos;

        boost::shared_ptr< std::vector< WVector3f > > eX( new std::vector< WVector3f >() );
        boost::shared_ptr< std::vector< WVector3f > > eY( new std::vector< WVector3f >() );
        boost::shared_ptr< std::vector< WVector3f > > eZ( new std::vector< WVector3f >() );
        float* eVec;

        fiffunits_t fiffUnit;

        for( size_t chan = 0; chan < dummy->getNrChans(); ++chan )
        {
            if( measinfo_in.GetLFChannelInfo()[chan]->GetKind() == mod )
            {
                if( mod == 1 )
                {
                    // TODO convert to millimeter?
                    eVec = measinfo_in.GetLFChannelInfo()[chan]->GetEx();
                    eX->push_back( WVector3f( eVec[0], eVec[1], eVec[2] ) );

                    eVec = measinfo_in.GetLFChannelInfo()[chan]->GetEy();
                    eY->push_back( WVector3f( eVec[0], eVec[1], eVec[2] ) );

                    eVec = measinfo_in.GetLFChannelInfo()[chan]->GetEz();
                    eZ->push_back( WVector3f( eVec[0], eVec[1], eVec[2] ) );
                }
                // Collect positions for EEG and MEG
                if( mod == 1 || mod == 2 )
                {
                    const float scale = 1000; // convert to millimeter
                    pos = measinfo_in.GetLFChannelInfo()[chan]->GetR0();
                    positions->push_back( WPosition( pos[0] * scale, pos[1] * scale, pos[2] * scale ) );
                }
                // Collect general data
                data->push_back( dummy->getData()[chan] );
                fiffUnit = measinfo_in.GetLFChannelInfo()[chan]->GetUnit();
            }
        }

        // Set general data
        emd->setSampFreq( dummy->getSampFreq() );
        emd->setAnalogHighPass( dummy->getAnalogHighPass() );
        emd->setAnalogLowPass( dummy->getAnalogLowPass() );
        emd->setLineFreq( dummy->getLineFreq() );
        emd->setChanUnitExp( LaBP::WEExponent::BASE ); // BASE because scaleFactor was multiplied
        emd->setChanUnit( getChanUnit( fiffUnit ) );
        emd->setData( data );

        switch( emd->getModalityType() )
        {
            case LaBP::WEModalityType::EEG: // Set specific EEG data
            {
                LaBP::WDataSetEMMEEG::SPtr eeg = boost::shared_dynamic_cast< LaBP::WDataSetEMMEEG >( emd );
                eeg->setChannelPositions3d( positions );
                wlog::debug( CLASS ) << "EEG positions: " << positions->size();
                break;
            }
            case LaBP::WEModalityType::MEG: // Set specific MEG data
            {
                LaBP::WDataSetEMMMEG::SPtr meg = boost::shared_dynamic_cast< LaBP::WDataSetEMMMEG >( emd );
                meg->setChannelPositions3d( positions );
                meg->setEx( eX );
                meg->setEx( eY );
                meg->setEx( eZ );
                wlog::debug( CLASS ) << "MEG positions: " << meg->getChannelPositions3d()->size();
                break;
            }
            default:
                // No specific data to set
                break;
        }

        out->addModality( emd );
    }

    // Create event/stimulus channel
    LFEvents events = measinfo_in.GetLFEvents();
    LaBP::WDataSetEMM::EChannelT eventData_out;
    std::vector< double > eventData_in;
    for( std::vector< int32_t >::iterator chan = events.GetEventChannels().begin(); chan != events.GetEventChannels().end();
                    ++chan )
    {
        wlog::debug( CLASS ) << "Event channel: " << *chan;
        eventData_out = LaBP::WDataSetEMM::EChannelT();
        eventData_in = dummy->getData()[*chan - 1]; // TODO LFEvents counts from 1 ?
        for( size_t i = 0; i < eventData_in.size(); ++i )
            eventData_out.push_back( ( LaBP::WDataSetEMM::EventT )eventData_in[i] );
        out->addEventChannel( eventData_out );
    }

    // Some debug out
    wlog::debug( CLASS ) << "LaBP::EMM data:";
    wlog::debug( CLASS ) << "\tLaBP::EMM::Event channels=" << out->getEventChannelCount();
    for( size_t mod = 0; mod < out->getModalityList().size(); ++mod )
    {
        wlog::debug( CLASS ) << "\tLaBP::EMM::EMD type=" << out->getModalityList()[mod]->getModalityType() << ", channels="
                        << out->getModalityList()[mod]->getNrChans();
    }

//    if( ret == rc_normal )
//        out->fireDataUpdateEvent();
    return getReturnCode( ret );
}

WLReaderFIFF::ReturnCode::Enum WLReaderFIFF::Read( LaBP::WDataSetEMMSubject::SPtr out )
{
    LFSubject data;
    returncode_t ret = LFInterface::fiffRead( data, m_fname.data() );
    if( ret != rc_normal )
        return getReturnCode( ret );
    out->setName( data.GetFirstName() + "" + data.GetMiddleName() + "" + data.GetLastName() );
    out->setHisId( data.GetHIS_ID() );
    out->setHeight( data.GetHeight() );
    out->setWeight( data.GetWeight() );
    out->setComment( data.GetComment() );
    switch( data.GetSex() )
    {
        case LFSubject::sex_m:
            out->setSex( LaBP::WESex::MALE );
            break;
        case LFSubject::sex_f:
            out->setSex( LaBP::WESex::FEMALE );
            break;
        default:
            out->setSex( LaBP::WESex::OTHER );
            break;
    }
    switch( data.GetHand() )
    {
        case LFSubject::hand_right:
            out->setHand( LaBP::WEHand::RIGHT );
            break;
        case LFSubject::hand_left:
            out->setHand( LaBP::WEHand::LEFT );
            break;
        default:
            out->setHand( LaBP::WEHand::BOTH );
            break;
    }
    //TODO(Evfimevskiy): data.GetBirthday();
    return getReturnCode( ret );
}

WLReaderFIFF::ReturnCode::Enum WLReaderFIFF::Read( std::vector< std::vector< double > >& out )
{
    LFRawData data;
    returncode_t ret = LFInterface::fiffRead( data, m_fname.data() );
    if( ret != rc_normal )
        return getReturnCode( ret );
    LFArrayPtr< LFDataBuffer >& rawdatabuffers_in = data.GetLFDataBuffer();
    size_t nBuffers_in = rawdatabuffers_in.size();
    out.resize( nBuffers_in );
    for( size_t i = 0; i < nBuffers_in; i++ )
    {
        LFDataBuffer* pBuf = rawdatabuffers_in[i];
        size_t nValues = pBuf->GetSize();
        out[i].resize( nValues );
        for( size_t j = 0; j < nValues; j++ )
        {
            switch( pBuf->GetDataType() )
            {
                case LFDataBuffer::dt_int16:
                    out[i][j] = pBuf->GetBufferInt16()->at( j );
                    break;
                case LFDataBuffer::dt_int32:
                    out[i][j] = pBuf->GetBufferInt32()->at( j );
                    break;
                case LFDataBuffer::dt_float:
                    out[i][j] = pBuf->GetBufferFloat()->at( j );
                    break;
                default:
                    // LFDataBuffer::dt_unknown
                    break;
            }
        }
    }
//    for( size_t i = 0; i < nBuffers_in; i++ ) printf("\nBuffer %d, value[0]==%lf",i,out[i][0]);//dbg
    return getReturnCode( ret );
//  return rc_normal; //dbg
}

WLReaderFIFF::ReturnCode::Enum WLReaderFIFF::getReturnCode( returncode_t rc )
{
    switch( rc )
    {
        case rc_normal:
            return ReturnCode::SUCCESS;
        case rc_error_file_open:
            return ReturnCode::ERROR_FOPEN;
        case rc_error_file_read:
            return ReturnCode::ERROR_FREAD;
        case rc_error_unknown:
            return ReturnCode::ERROR_UNKNOWN;
        default:
            return ReturnCode::ERROR_UNKNOWN;
    }

}

LaBP::WEUnit::Enum WLReaderFIFF::getChanUnit( fiffunits_t unit )
{
    switch( unit )
    {
        case unit_V:
            return LaBP::WEUnit::VOLT;
        case unit_T:
            return LaBP::WEUnit::TESLA;
        case unit_T_m:
            return LaBP::WEUnit::TESLA_PER_METER;
        default:
            return LaBP::WEUnit::UNKNOWN_UNIT;
    }
}
