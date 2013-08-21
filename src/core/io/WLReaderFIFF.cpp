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

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMEnumTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDECG.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDMEG.h"

#include "WLReaderFIFF.h"

using namespace LaBP;

const std::string WLReaderFIFF::CLASS = "WLReaderFIFF";

WLReaderFIFF::WLReaderFIFF( std::string fname ) :
                WLReader( fname )
{
    wlog::debug( CLASS ) << "file: " << fname;
}

WLReaderFIFF::ReturnCode::Enum WLReaderFIFF::Read( WLEMMeasurement::SPtr out )
{
    LFData data;
    returncode_t ret = LFInterface::fiffRead( data, m_fname.data() );
    if( ret != rc_normal )
        return getReturnCode( ret );

    // Set subject information
    LaBP::WLEMMSubject::SPtr subject_out( new LaBP::WLEMMSubject() );
    ReturnCode::Enum rc = Read( subject_out );
    if( rc != ReturnCode::SUCCESS )
        return rc;
    out->setSubject( subject_out );

    // Create temporary EMMEMD
    WLEMDEEG::SPtr dummy( new WLEMDEEG() );
    LFMeasurementInfo& measinfo_in = data.GetLFMeasurement().GetLFMeasurementInfo();
    int32_t nChannels = measinfo_in.GetNumberOfChannels();
    wlog::debug( CLASS ) << "Channels: " << nChannels;
    dummy->setSampFreq( measinfo_in.GetSamplingFrequency() );
    dummy->setAnalogHighPass( measinfo_in.GetHighpass() );
    dummy->setAnalogLowPass( measinfo_in.GetLowpass() );
    dummy->setLineFreq( measinfo_in.GetLineFreq() );

    // Read Isotrak
    LFIsotrak& isotrak = measinfo_in.GetLFIsotrak();
    LFArrayPtr< LFDigitisationPoint > &digPoints = isotrak.GetLFDigitisationPoint();
    boost::shared_ptr< std::vector< WVector3f > > itPos( new std::vector< WVector3f >() );
    for( LFArrayPtr< LFDigitisationPoint >::size_type i = 0; i < digPoints.size(); ++i )
    {
        itPos->push_back( WVector3f( digPoints[i]->GetRr()[0], digPoints[i]->GetRr()[1], digPoints[i]->GetRr()[2] ) );
    }
    subject_out->setIsotrak( itPos );
    wlog::debug( CLASS ) << "Isotrak size: " << itPos->size();

    // Read raw data
    LFRawData& rawdata_in = data.GetLFMeasurement().GetLFRawData();
    LFArrayPtr< LFDataBuffer > &rawdatabuffers_in = rawdata_in.GetLFDataBuffer();
    size_t nBuffers_in = rawdatabuffers_in.size();
    size_t nBuffers_out = 0;
    for( size_t i = 0; i < nBuffers_in; ++i )
        nBuffers_out += rawdatabuffers_in[i]->GetSize();
    nBuffers_out /= nChannels;
    WLEMData::DataT rawdatabuffers_out( nChannels, nBuffers_out );
    int32_t current_channel = 0, current_buffer_out = 0;
    LFArrayPtr< LFChannelInfo > &channelInfos = measinfo_in.GetLFChannelInfo();
    double scaleFactor;
    for( size_t i = 0; i < nBuffers_in; ++i )
    {
        LFDataBuffer* pBuf = rawdatabuffers_in[i];
        size_t nValues = pBuf->GetSize();
        for( size_t j = 0; j < nValues; ++j )
        {
            scaleFactor = channelInfos[current_channel]->GetRange() * channelInfos[current_channel]->GetCal();
            switch( pBuf->GetDataType() )
            {
                case LFDataBuffer::dt_int16:
                    rawdatabuffers_out( current_channel, current_buffer_out ) = ( WLEMData::ScalarT )pBuf->GetBufferInt16()->at(
                                    j ) * scaleFactor;
                    break;
                case LFDataBuffer::dt_int32:
                    rawdatabuffers_out( current_channel, current_buffer_out ) = ( WLEMData::ScalarT )pBuf->GetBufferInt32()->at(
                                    j ) * scaleFactor;
                    break;
                case LFDataBuffer::dt_float:
                    rawdatabuffers_out( current_channel, current_buffer_out ) = ( WLEMData::ScalarT )pBuf->GetBufferFloat()->at(
                                    j ) * scaleFactor;
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

    WLEMData::DataSPtr rawdatabuffers_out_ptr( new WLEMData::DataT( rawdatabuffers_out ) );
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
        WLEMData::SPtr emd;
        // See FIFF PDF B.3 Channel Types
        switch( mod )
        {
            case 1: // MEG channel
                wlog::debug( CLASS ) << "Creating MEG modality ...";
                emd.reset( new WLEMDMEG() );
                break;
            case 2: // EEG channel
                wlog::debug( CLASS ) << "Creating EEG modality ...";
                emd.reset( new WLEMDEEG() );
                break;
//            case 3: // Stimulus channel
            case 202: // EOG channel
                wlog::debug( CLASS ) << "Creating EOG modality ...";
                emd.reset( new WLEMDEOG() );
                break;
            case 402: // ECG channel
                wlog::debug( CLASS ) << "Creating ECG modality ...";
                emd.reset( new WLEMDECG() );
                break;
            default:
                wlog::debug( CLASS ) << "Skip modality type: " << mod;
                continue;
        }

        // Collect data: measurement, positions, base vectors
        // TODO(pieloth): set channels size

        boost::shared_ptr< std::vector< WPosition > > positions( new std::vector< WPosition >() );
        float* pos;

        boost::shared_ptr< std::vector< WVector3f > > eX( new std::vector< WVector3f >() );
        boost::shared_ptr< std::vector< WVector3f > > eY( new std::vector< WVector3f >() );
        boost::shared_ptr< std::vector< WVector3f > > eZ( new std::vector< WVector3f >() );
        float* eVec;

        fiffunits_t fiffUnit;

        size_t modChan = 0;
        WLEMData::DataT dataTmp( rawdatabuffers_out_ptr->rows(), rawdatabuffers_out_ptr->cols() );
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

                    // Check sequence
                    const int32_t coil_type = measinfo_in.GetLFChannelInfo()[chan]->GetCoilType();
                    // TODO(pieloth): check meg coil order
                    if( modChan > 1 && ( modChan - 2 ) % 3 == 0 )
                    {
                        WAssert( coil_type == 3021 || coil_type == 3022 || coil_type == 3024,
                                        "Wrong order! Coil type should be magentometer!" );
                    }
                    else
                    {
                        WAssert( coil_type == 3012 || coil_type == 3013 || coil_type == 3014,
                                        "Wrong order! Coil type should be gradiometer!" );
                    }
                }
                // Collect positions for EEG and MEG
                if( mod == 1 || mod == 2 )
                {
                    pos = measinfo_in.GetLFChannelInfo()[chan]->GetR0();
                    positions->push_back( WPosition( pos[0], pos[1], pos[2] ) );
                }

                // Collect general data
                dataTmp.row( modChan++ ) = ( dummy->getData().row( chan ) );
                fiffUnit = measinfo_in.GetLFChannelInfo()[chan]->GetUnit();
            }
        }

        WLEMData::DataSPtr data( new WLEMData::DataT( dataTmp.block( 0, 0, modChan, dataTmp.cols() ) ) );

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
                WLEMDEEG::SPtr eeg = emd->getAs< WLEMDEEG >();
                eeg->setChannelPositions3d( positions );
                wlog::debug( CLASS ) << "EEG positions: " << positions->size();
                break;
            }
            case LaBP::WEModalityType::MEG: // Set specific MEG data
            {
                WLEMDMEG::SPtr meg = emd->getAs< WLEMDMEG >();
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
    WLEMMeasurement::EChannelT eventData_out;
    WLEMData::ChannelT eventData_in;
    for( std::vector< int32_t >::iterator chan = events.GetEventChannels().begin(); chan != events.GetEventChannels().end();
                    ++chan )
    {
        wlog::debug( CLASS ) << "Event channel: " << *chan;
        eventData_out = WLEMMeasurement::EChannelT();
        eventData_in = dummy->getData().row( *chan - 1 ); // TODO LFEvents counts from 1 ?
        for( size_t i = 0; i < eventData_in.size(); ++i )
            eventData_out.push_back( ( WLEMMeasurement::EventT )eventData_in( i ) );
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

WLReaderFIFF::ReturnCode::Enum WLReaderFIFF::Read( LaBP::WLEMMSubject::SPtr out )
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
