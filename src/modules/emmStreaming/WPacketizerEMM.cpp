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

#include "../emmStreaming/WPacketizerEMM.h"

#include <set>
#include <string>
#include <vector>

#include <core/common/WLogger.h>


const std::string WPacketizerEMM::CLASS = "WPacketizerEMM";

WPacketizerEMM::WPacketizerEMM( WLEMMeasurement::ConstSPtr data, size_t blockSize ) :
                WPacketizer( data, blockSize ), m_blockCount( 0 )
{
    m_hasData = false;
    const std::set< WLEModality::Enum > mods = data->getModalityTypes();
    std::set< WLEModality::Enum >::const_iterator it;
    for( it = mods.begin(); it != mods.end(); ++it )
    {
        m_emds.push_back( data->getModality( *it ) );
        m_hasData = m_hasData || data->getModality( *it )->getSamplesPerChan() > 0;
    }

    m_events = data->getEventChannels();
}
WPacketizerEMM::~WPacketizerEMM()
{
}

bool WPacketizerEMM::hasNext() const
{
    return m_hasData;
}

WLEMMeasurement::SPtr WPacketizerEMM::next()
{
    WLEMMeasurement::SPtr emmPacket = m_data->clone();
    m_hasData = false;

    float smplFrq;
    WLSampleNrT samples = 0;
    WLSampleNrT samplesOffset;
    // clone each modality
    std::vector< WLEMData::ConstSPtr >::const_iterator itEmd;
    for( itEmd = m_emds.begin(); itEmd != m_emds.end(); ++itEmd )
    {
        smplFrq = ( *itEmd )->getSampFreq();
        samples = smplFrq * m_blockSize / 1000;
        samplesOffset = m_blockCount * samples;
        WLEMData::SPtr emdPacket = ( *itEmd )->clone();
        WLEMData::DataSPtr emdData( new WLEMData::DataT( ( *itEmd )->getNrChans(), samples ) );
        const WLEMData::DataT orgData = ( *itEmd )->getData();

        // copy each channel
        for( WLChanIdxT chan = 0; chan < ( *itEmd )->getNrChans(); ++chan )
        {
            for( WLSampleIdxT sample = 0; sample < samples && ( samplesOffset + sample ) < orgData.cols(); ++sample )
            {
                ( *emdData )( chan, sample ) = orgData( chan, samplesOffset + sample );
            }
        }

        emdPacket->setData( emdData );
        emmPacket->addModality( emdPacket );

        if( ( *itEmd )->getNrChans() > 0 )
        {
            wlog::debug( CLASS ) << "emdPacket type: " << WLEModality::name( emdPacket->getModalityType() ) << " size: "
                            << emdPacket->getData().cols();

            // set termination condition
            m_hasData = m_hasData || samplesOffset + samples < static_cast< size_t >( orgData.cols() );
        }
    }

    // copy event channels
    samplesOffset = m_blockCount * samples; // Using blockSize/samplFreq of last modality
    WLEMMeasurement::EDataT::const_iterator itEChans;
    for( itEChans = m_events->begin(); itEChans != m_events->end(); ++itEChans )
    {
        WLEMMeasurement::EChannelT eChan;
        eChan.reserve( samples );
        for( WLSampleIdxT event = 0; event < samples && ( samplesOffset + event ) < ( *itEChans ).size(); ++event )
        {
            eChan.push_back( ( *itEChans )[samplesOffset + event] );
        }
        emmPacket->addEventChannel( eChan );

        wlog::debug( CLASS ) << "emmPacket event size: " << eChan.size();
    }

    ++m_blockCount;

    return emmPacket;
}

