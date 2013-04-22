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
#include <core/dataHandler/WDataSetEMMEMD.h>

float LaBP::WDataSetEMMEMD::m_lineFreq;

LaBP::WDataSetEMMEMD::WDataSetEMMEMD()
{
    m_data.reset( new DataT() );
    m_chanNames.reset( new std::vector< std::string >() );
}

LaBP::WDataSetEMMEMD::WDataSetEMMEMD( const WDataSetEMMEMD& emd )
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

LaBP::WDataSetEMMEMD::~WDataSetEMMEMD()
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

LaBP::WDataSetEMMEMD::DataT& LaBP::WDataSetEMMEMD::getData() const
{
    return *m_data;
}

void LaBP::WDataSetEMMEMD::setData( boost::shared_ptr< DataT > data )
{
    m_data = data;
}

void LaBP::WDataSetEMMEMD::addSample( double value )
{
    std::vector< double > channel;
    channel.push_back( value );
    m_data->push_back( channel );
}

float LaBP::WDataSetEMMEMD::getAnalogHighPass() const
{
    return m_analogHighPass;
}

float LaBP::WDataSetEMMEMD::getAnalogLowPass() const
{
    return m_analogLowPass;
}

std::vector< std::string >& LaBP::WDataSetEMMEMD::getChanNames() const
{
    return *m_chanNames;
}

LaBP::WEUnit::Enum LaBP::WDataSetEMMEMD::getChanUnit() const
{
    return m_chanUnit;
}

LaBP::WEExponent::Enum LaBP::WDataSetEMMEMD::getChanUnitExp() const
{
    return m_chanUnitExp;
}

LaBP::WECoordSystemName::Enum LaBP::WDataSetEMMEMD::getCoordSystem() const
{
    return m_CoordSystem;
}

uint32_t LaBP::WDataSetEMMEMD::getDataBuffSizePerChan() const
{
    return m_dataBuffSizePerChan;
}

uint32_t LaBP::WDataSetEMMEMD::getDataOffsetIdx() const
{
    return m_dataOffsetIdx;
}

float LaBP::WDataSetEMMEMD::getLineFreq()
{
    return m_lineFreq;
}

std::string LaBP::WDataSetEMMEMD::getMeasurementDeviceName() const
{
    return m_measurementDeviceName;
}

size_t LaBP::WDataSetEMMEMD::getNrChans() const
{
    return m_data->size();
}

size_t LaBP::WDataSetEMMEMD::getSamplesPerChan() const
{
    if( m_data && m_data->size() > 0 )
    {
        return m_data->front().size();
    }
    return 0;
}

uint16_t *LaBP::WDataSetEMMEMD::getOrigIdx() const
{
    return m_origIdx;
}

float LaBP::WDataSetEMMEMD::getSampFreq() const
{
    return m_sampFreq;
}

float LaBP::WDataSetEMMEMD::getLength() const
{
    return getSamplesPerChan() / m_sampFreq;
}

void LaBP::WDataSetEMMEMD::setAnalogHighPass( float analogHighPass )
{
    m_analogHighPass = analogHighPass;
}

void LaBP::WDataSetEMMEMD::setAnalogLowPass( float analogLowPass )
{
    m_analogLowPass = analogLowPass;
}

void LaBP::WDataSetEMMEMD::setChanNames( boost::shared_ptr< std::vector< std::string > > chanNames )
{
    m_chanNames = chanNames;
}

void LaBP::WDataSetEMMEMD::setChanUnit( LaBP::WEUnit::Enum chanUnit )
{
    m_chanUnit = chanUnit;
}

void LaBP::WDataSetEMMEMD::setChanUnitExp( LaBP::WEExponent::Enum chanUnitExp )
{
    m_chanUnitExp = chanUnitExp;
}

void LaBP::WDataSetEMMEMD::setCoordSystem( LaBP::WECoordSystemName::Enum coordSystem )
{
    m_CoordSystem = coordSystem;
}

void LaBP::WDataSetEMMEMD::setDataBuffSizePerChan( uint32_t dataBuffSizePerChan )
{
    m_dataBuffSizePerChan = dataBuffSizePerChan;
}

void LaBP::WDataSetEMMEMD::setDataOffsetIdx( uint32_t dataOffsetIdx )
{
    m_dataOffsetIdx = dataOffsetIdx;
}

void LaBP::WDataSetEMMEMD::setLineFreq( float lineFreq )
{
    m_lineFreq = lineFreq;
}

void LaBP::WDataSetEMMEMD::setMeasurementDeviceName( std::string measurementDeviceName )
{
    m_measurementDeviceName = measurementDeviceName;
}

void LaBP::WDataSetEMMEMD::setOrigIdx( uint16_t *origIdx )
{
    m_origIdx = origIdx;
}

void LaBP::WDataSetEMMEMD::setSampFreq( float sampFreq )
{
    m_sampFreq = sampFreq;
}

std::string LaBP::WDataSetEMMEMD::channelToString( const ChannelT& data, size_t maxSamples )
{
    const size_t nbSmp = data.size();
    std::stringstream ss;
    for( size_t j = 0; j < maxSamples && j < nbSmp; ++j )
    {
        ss << data[j] << " ";
    }
    return ss.str();
}

std::string LaBP::WDataSetEMMEMD::dataToString( const DataT& data, size_t maxChannels, size_t maxSamples )
{
    const size_t nbChan = data.size();

    std::stringstream ss;
    for( size_t i = 0; i < maxChannels && i < nbChan; ++i )
    {
        ss << "Channel " << i << ": " << channelToString( data[i], maxSamples ) << std::endl;
    }
    return ss.str();
}

