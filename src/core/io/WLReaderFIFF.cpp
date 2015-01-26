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
#include <set>
#include <string>
#include <vector>

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

#include "core/container/WLArrayList.h"
#include "core/container/WLList.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDECG.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/emd/WLEMDRaw.h"
#include "core/data/enum/WLEModality.h"

#include "core/dataFormat/fiff/WLFiffChType.h"
#include "core/dataFormat/fiff/WLFiffCoilType.h"
#include "core/dataFormat/fiff/WLFiffUnit.h"
#include "core/dataFormat/fiff/WLFiffUnitMultiplier.h"

#include "WLReaderFIFF.h"

const std::string WLReaderFIFF::CLASS = "WLReaderFIFF";

WLReaderFIFF::WLReaderFIFF( std::string fname ) :
                WLReaderGeneric< WLEMMeasurement::SPtr >( fname )
{
    wlog::debug( CLASS ) << "file: " << fname;
}

WLIOStatus::IOStatusT WLReaderFIFF::read( WLEMMeasurement::SPtr* const out )
{
    LFData data;
    returncode_t ret = LFInterface::fiffRead( data, m_fname.data() );
    if( ret != rc_normal )
        return getReturnCode( ret );

    // Set subject information
    WLEMMSubject::SPtr subject_out( new WLEMMSubject() );
    WLIOStatus::IOStatusT rc = read( &subject_out );
    if( rc != WLIOStatus::SUCCESS )
        return rc;
    ( *out )->setSubject( subject_out );

    // Create temporary EMMEMD
    WLEMDRaw::SPtr emdRaw( new WLEMDRaw() );
    LFMeasurementInfo& measinfo_in = data.GetLFMeasurement().GetLFMeasurementInfo();
    WLFiffLib::nchan_t nChannels = measinfo_in.GetNumberOfChannels();
    wlog::debug( CLASS ) << "Channels: " << nChannels;
    // Frequencies are in Hz, see Functional Image File Format, Appendix C.3 Common data tags
    emdRaw->setSampFreq( measinfo_in.GetSamplingFrequency() * WLUnits::Hz );
    emdRaw->setAnalogHighPass( measinfo_in.GetHighpass() * WLUnits::Hz );
    emdRaw->setAnalogLowPass( measinfo_in.GetLowpass() * WLUnits::Hz );
    emdRaw->setLineFreq( measinfo_in.GetLineFreq() * WLUnits::Hz );

    // Read Isotrak
    LFIsotrak& isotrak = measinfo_in.GetLFIsotrak();
    LFArrayPtr< LFDigitisationPoint > &digPoints = isotrak.GetLFDigitisationPoint();
    WLArrayList< WVector3f >::SPtr itPos( new WLArrayList< WVector3f >() );
    WLList< WLDigPoint >::SPtr digPointsOut( new WLList< WLDigPoint >() );
    for( LFArrayPtr< LFDigitisationPoint >::size_type i = 0; i < digPoints.size(); ++i )
    {
        itPos->push_back( WVector3f( digPoints[i]->GetRr()[0], digPoints[i]->GetRr()[1], digPoints[i]->GetRr()[2] ) );
        const WPosition pos( digPoints[i]->GetRr()[0], digPoints[i]->GetRr()[1], digPoints[i]->GetRr()[2] );
        const WLDigPoint digPoint( pos, digPoints[i]->GetKind(), digPoints[i]->GetIdent() );
        digPointsOut->push_back( digPoint );
    }

    subject_out->setIsotrak( itPos );
    ( *out )->setDigPoints( digPointsOut );
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
    WLFiffLib::ichan_t current_channel = 0, current_buffer_out = 0;
    LFArrayPtr< LFChannelInfo > &channelInfos = measinfo_in.GetLFChannelInfo();
    double scaleFactor;
    for( size_t i = 0; i < nBuffers_in; ++i )
    {
        LFDataBuffer* pBuf = rawdatabuffers_in[i];
        size_t nValues = pBuf->GetSize();
        for( size_t j = 0; j < nValues; ++j )
        {
            // scaleFactor: see FIFF spec. 1.3, 3.5.4 Raw data files, p. 15
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
                    wlog::warn( CLASS ) << "Unknown data type: " << pBuf->GetDataType();
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
    emdRaw->setData( rawdatabuffers_out_ptr );

    // Collect available modalities and coils
    std::set< WLFiffLib::ch_type_t > modalities;
    for( WLFiffLib::nchan_t chan = 0; chan < nChannels; ++chan )
    {
        modalities.insert( measinfo_in.GetLFChannelInfo()[chan]->GetKind() );
    }

    // Create modality objects //
    WLFiffLib::ch_type_t mod;
    while( !modalities.empty() )
    {
        mod = *modalities.begin();
        modalities.erase( mod );
        WLEMData::SPtr emd;
        // See FIFF PDF B.3 Channel Types
        switch( mod )
        {
            case WLFiffLib::ChType::MAGN: // MEG channel
                wlog::debug( CLASS ) << "Creating MEG modality ...";
                emd.reset( new WLEMDMEG() );
                break;
            case WLFiffLib::ChType::EL: // EEG channel
                wlog::debug( CLASS ) << "Creating EEG modality ...";
                emd.reset( new WLEMDEEG() );
                break;
            case WLFiffLib::ChType::STIM: // Stimulus channel
                wlog::debug( CLASS ) << "Stim channel is processed later ...";
                continue;
            case WLFiffLib::ChType::EOG: // EOG channel
                wlog::debug( CLASS ) << "Creating EOG modality ...";
                emd.reset( new WLEMDEOG() );
                break;
            case WLFiffLib::ChType::ECG: // ECG channel
                wlog::debug( CLASS ) << "Creating ECG modality ...";
                emd.reset( new WLEMDECG() );
                break;
            default:
                wlog::debug( CLASS ) << "Skip modality type: " << mod;
                continue;
        }

        // Collect data: measurement, positions, base vectors
        // TODO(pieloth): set channels size
        std::vector< WLPositions::PositionT > posTmp;
        const float* pos;

        WLArrayList< WVector3f >::SPtr eX( new WLArrayList< WVector3f >() );
        WLArrayList< WVector3f >::SPtr eY( new WLArrayList< WVector3f >() );
        WLArrayList< WVector3f >::SPtr eZ( new WLArrayList< WVector3f >() );
        const float* eVec;

        WLFiffLib::unit_t fiffUnit = WLFiffLib::Unit::NONE;
        WLFiffLib::unitm_t fiffUnitMul = WLFiffLib::UnitMultiplier::NONE;

        size_t modChan = 0;
        WLEMData::DataT dataTmp( rawdatabuffers_out_ptr->rows(), rawdatabuffers_out_ptr->cols() );
        for( WLFiffLib::nchan_t chan = 0; chan < emdRaw->getNrChans(); ++chan )
        {
            if( measinfo_in.GetLFChannelInfo()[chan]->GetKind() == mod )
            {
                if( mod == WLFiffLib::ChType::MAGN )
                {
                    eVec = measinfo_in.GetLFChannelInfo()[chan]->GetEx();
                    eX->push_back( WVector3f( eVec[0], eVec[1], eVec[2] ) );

                    eVec = measinfo_in.GetLFChannelInfo()[chan]->GetEy();
                    eY->push_back( WVector3f( eVec[0], eVec[1], eVec[2] ) );

                    eVec = measinfo_in.GetLFChannelInfo()[chan]->GetEz();
                    eZ->push_back( WVector3f( eVec[0], eVec[1], eVec[2] ) );

                    // Check sequence
                    const WLFiffLib::coil_type_t coil_type = measinfo_in.GetLFChannelInfo()[chan]->GetCoilType();
                    // TODO(pieloth): check meg coil order
                    if( modChan > 1 && ( modChan - 2 ) % 3 == 0 )
                    {
                        WAssert(
                                        coil_type == WLFiffLib::CoilType::VV_MAG_W || coil_type == WLFiffLib::CoilType::VV_MAG_T1
                                                        || coil_type == WLFiffLib::CoilType::VV_MAG_T2
                                                        || coil_type == WLFiffLib::CoilType::VV_MAG_T3,
                                        "Wrong order! Coil type should be magentometer!" );
                    }
                    else
                    {
                        WAssert(
                                        coil_type == WLFiffLib::CoilType::VV_PLANAR_W
                                                        || coil_type == WLFiffLib::CoilType::VV_PLANAR_T1
                                                        || coil_type == WLFiffLib::CoilType::VV_PLANAR_T2
                                                        || coil_type == WLFiffLib::CoilType::VV_PLANAR_T3,
                                        "Wrong order! Coil type should be gradiometer!" );
                    }
                }
                // Collect positions for EEG and MEG
                if( mod == WLFiffLib::ChType::MAGN || mod == WLFiffLib::ChType::EL )
                {
                    pos = measinfo_in.GetLFChannelInfo()[chan]->GetR0();
                    posTmp.push_back( WLPositions::PositionT( pos[0], pos[1], pos[2] ) );
                }

                // Collect general data
                dataTmp.row( modChan++ ) = ( emdRaw->getData().row( chan ) );
                fiffUnit = static_cast< WLFiffLib::unit_t >( measinfo_in.GetLFChannelInfo()[chan]->GetUnit() );
                fiffUnitMul = static_cast< WLFiffLib::unitm_t >( measinfo_in.GetLFChannelInfo()[chan]->GetUnitMul() );
            }
        }

        WLEMData::DataSPtr data( new WLEMData::DataT( dataTmp.block( 0, 0, modChan, dataTmp.cols() ) ) );

        // Set general data
        emd->setSampFreq( emdRaw->getSampFreq() );
        emd->setAnalogHighPass( emdRaw->getAnalogHighPass() );
        emd->setAnalogLowPass( emdRaw->getAnalogLowPass() );
        emd->setLineFreq( emdRaw->getLineFreq() );

        // scaleFactor was multiplied, so data is in unit_mul - see FIFF spec. 1.3, Table A.3, p. 28)
        emd->setChanUnitExp( WLEExponent::fromFIFF( fiffUnitMul ) );
        emd->setChanUnit( WLEUnit::fromFIFF( fiffUnit ) );
        emd->setData( data );

        WLPositions::SPtr positions = WLPositions::instance();
        positions->resize( posTmp.size() );
        for( WLPositions::IndexT i = 0; i < positions->size(); ++i )
        {
            positions->data().col( i ) = posTmp.at( i );
        }

        switch( emd->getModalityType() )
        {
            case WLEModality::EEG: // Set specific EEG data
            {
                WLEMDEEG::SPtr eeg = emd->getAs< WLEMDEEG >();
                eeg->setChannelPositions3d( positions );
                wlog::debug( CLASS ) << "EEG positions: " << posTmp.size();
                break;
            }
            case WLEModality::MEG: // Set specific MEG data
            {
                WLEMDMEG::SPtr meg = emd->getAs< WLEMDMEG >();
                meg->setChannelPositions3d( positions );
                meg->setEx( eX );
                meg->setEy( eY );
                meg->setEz( eZ );
                wlog::debug( CLASS ) << "MEG positions: " << posTmp.size();
                break;
            }
            default:
                // No specific data to set
                break;
        }

        ( *out )->addModality( emd );
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
        eventData_in = emdRaw->getData().row( *chan - 1 ); // TODO(pieloth): LFEvents counts from 1 ?
        for( WLEMData::ChannelT::Index i = 0; i < eventData_in.size(); ++i )
        {
            eventData_out.push_back( ( WLEMMeasurement::EventT )eventData_in( i ) );
        }
        ( *out )->addEventChannel( eventData_out );
    }

    // Some debug out
    wlog::debug( CLASS ) << "LaBP::EMM data:";
    wlog::debug( CLASS ) << "\tLaBP::EMM::Event channels=" << ( *out )->getEventChannelCount();
    for( size_t mod = 0; mod < ( *out )->getModalityList().size(); ++mod )
    {
        wlog::debug( CLASS ) << "\tLaBP::EMM::EMD type=" << ( *out )->getModalityList()[mod]->getModalityType() << ", channels="
                        << ( *out )->getModalityList()[mod]->getNrChans();
    }

    return getReturnCode( ret );
}

WLIOStatus::IOStatusT WLReaderFIFF::read( WLEMMSubject::SPtr* const out )
{
    LFSubject data;
    returncode_t ret = LFInterface::fiffRead( data, m_fname.data() );
    if( ret != rc_normal )
        return getReturnCode( ret );
    ( *out )->setName( data.GetFirstName() + "" + data.GetMiddleName() + "" + data.GetLastName() );
    ( *out )->setHisId( data.GetHIS_ID() );
    ( *out )->setComment( data.GetComment() );
    return getReturnCode( ret );
}

WLIOStatus::IOStatusT WLReaderFIFF::getReturnCode( returncode_t rc )
{
    switch( rc )
    {
        case rc_normal:
            return WLIOStatus::SUCCESS;
        case rc_error_file_open:
            return WLIOStatus::ERROR_FOPEN;
        case rc_error_file_read:
            return WLIOStatus::ERROR_FREAD;
        case rc_error_unknown:
            return WLIOStatus::ERROR_UNKNOWN;
        default:
            return WLIOStatus::ERROR_UNKNOWN;
    }
}
