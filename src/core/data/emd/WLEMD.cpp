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

#include <sstream>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>

#include "WLEMD.h"

float LaBP::WLEMD::m_lineFreq;

LaBP::WLEMD::WLEMD()
{
    m_data.reset( new DataT() );
    m_chanNames.reset( new std::vector< std::string >() );
}

LaBP::WLEMD::WLEMD( const WLEMD& emd )
{
    // C++11 supports "delegating constructors". So default initialization could be moved to default constructor.
    m_data.reset( new DataT() );

    m_chanNames = emd.m_chanNames;
    if( m_chanNames->empty() )
    {
        wlog::info( "WDataSetEMMEMD" ) << "No channel names available! Channels will be numbered.";
        const size_t chanSize = emd.getNrChans();
        for( size_t i = 0; i < chanSize; ++i )
        {
            m_chanNames->push_back( boost::lexical_cast< std::string >( i ) );
        }
    }
    m_measurementDeviceName.assign( emd.getMeasurementDeviceName() );
    m_sampFreq = emd.getSampFreq();
    m_chanUnit = emd.getChanUnit();
    m_chanUnitExp = emd.getChanUnitExp();
    m_analogHighPass = emd.getAnalogHighPass();
    m_analogLowPass = emd.getAnalogLowPass();
    m_CoordSystem = emd.getCoordSystem();
    m_dataBuffSizePerChan = emd.getDataBuffSizePerChan(); // TODO check
    m_dataOffsetIdx = emd.getDataOffsetIdx();
}

LaBP::WLEMD::~WLEMD()
{
}

//template< typename EMD >
//boost::shared_ptr< EMD > LaBP::WDataSetEMMEMD::getAs()
//{
//    return boost::dynamic_pointer_cast< EMD >( shared_from_this() );
//}

//template< typename EMD >
//boost::shared_ptr< const EMD > LaBP::WDataSetEMMEMD::getAs( LaBP::WDataSetEMMEMD::ConstSPtr emd )
//{
//
//    return boost::dynamic_pointer_cast< const EMD >( emd );
//}

LaBP::WLEMD::DataT& LaBP::WLEMD::getData() const
{
    return *m_data;
}

void LaBP::WLEMD::setData( boost::shared_ptr< DataT > data )
{
    m_data = data;
}

void LaBP::WLEMD::addSample( double value )
{
    std::vector< double > channel;
    channel.push_back( value );
    m_data->push_back( channel );
}

float LaBP::WLEMD::getAnalogHighPass() const
{
    return m_analogHighPass;
}

float LaBP::WLEMD::getAnalogLowPass() const
{
    return m_analogLowPass;
}

std::vector< std::string >& LaBP::WLEMD::getChanNames() const
{
    return *m_chanNames;
}

LaBP::WEUnit::Enum LaBP::WLEMD::getChanUnit() const
{
    return m_chanUnit;
}

LaBP::WEExponent::Enum LaBP::WLEMD::getChanUnitExp() const
{
    return m_chanUnitExp;
}

LaBP::WECoordSystemName::Enum LaBP::WLEMD::getCoordSystem() const
{
    return m_CoordSystem;
}

uint32_t LaBP::WLEMD::getDataBuffSizePerChan() const
{
    return m_dataBuffSizePerChan;
}

uint32_t LaBP::WLEMD::getDataOffsetIdx() const
{
    return m_dataOffsetIdx;
}

float LaBP::WLEMD::getLineFreq()
{
    return m_lineFreq;
}

std::string LaBP::WLEMD::getMeasurementDeviceName() const
{
    return m_measurementDeviceName;
}

size_t LaBP::WLEMD::getNrChans() const
{
    return m_data->size();
}

size_t LaBP::WLEMD::getSamplesPerChan() const
{
    if( m_data && m_data->size() > 0 )
    {
        return m_data->front().size();
    }
    return 0;
}

uint16_t *LaBP::WLEMD::getOrigIdx() const
{
    return m_origIdx;
}

float LaBP::WLEMD::getSampFreq() const
{
    return m_sampFreq;
}

float LaBP::WLEMD::getLength() const
{
    return getSamplesPerChan() / m_sampFreq;
}

void LaBP::WLEMD::setAnalogHighPass( float analogHighPass )
{
    m_analogHighPass = analogHighPass;
}

void LaBP::WLEMD::setAnalogLowPass( float analogLowPass )
{
    m_analogLowPass = analogLowPass;
}

void LaBP::WLEMD::setChanNames( boost::shared_ptr< std::vector< std::string > > chanNames )
{
    m_chanNames = chanNames;
}

void LaBP::WLEMD::setChanUnit( LaBP::WEUnit::Enum chanUnit )
{
    m_chanUnit = chanUnit;
}

void LaBP::WLEMD::setChanUnitExp( LaBP::WEExponent::Enum chanUnitExp )
{
    m_chanUnitExp = chanUnitExp;
}

void LaBP::WLEMD::setCoordSystem( LaBP::WECoordSystemName::Enum coordSystem )
{
    m_CoordSystem = coordSystem;
}

void LaBP::WLEMD::setDataBuffSizePerChan( uint32_t dataBuffSizePerChan )
{
    m_dataBuffSizePerChan = dataBuffSizePerChan;
}

void LaBP::WLEMD::setDataOffsetIdx( uint32_t dataOffsetIdx )
{
    m_dataOffsetIdx = dataOffsetIdx;
}

void LaBP::WLEMD::setLineFreq( float lineFreq )
{
    m_lineFreq = lineFreq;
}

void LaBP::WLEMD::setMeasurementDeviceName( std::string measurementDeviceName )
{
    m_measurementDeviceName = measurementDeviceName;
}

void LaBP::WLEMD::setOrigIdx( uint16_t *origIdx )
{
    m_origIdx = origIdx;
}

void LaBP::WLEMD::setSampFreq( float sampFreq )
{
    m_sampFreq = sampFreq;
}

std::string LaBP::WLEMD::channelToString( const ChannelT& data, size_t maxSamples )
{
    const size_t nbSmp = data.size();
    std::stringstream ss;
    for( size_t j = 0; j < maxSamples && j < nbSmp; ++j )
    {
        ss << data[j] << " ";
    }
    return ss.str();
}

std::string LaBP::WLEMD::dataToString( const DataT& data, size_t maxChannels, size_t maxSamples )
{
    const size_t nbChan = data.size();

    std::stringstream ss;
    for( size_t i = 0; i < maxChannels && i < nbChan; ++i )
    {
        ss << "Channel " << i << ": " << channelToString( data[i], maxSamples ) << std::endl;
    }
    return ss.str();
}

