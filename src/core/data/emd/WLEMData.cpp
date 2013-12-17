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
#include <ostream> // std::endl

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>

#include "WLEMData.h"

using namespace LaBP;

const std::string WLEMData::CLASS = "WLEMData";

WLEMData::WLEMData() :
                boost::enable_shared_from_this< WLEMData >()
{
    m_data.reset( new DataT() );
    m_chanNames = WLArrayList< std::string >::instance();

    m_badChannels.reset( new ChannelList() );
}

WLEMData::WLEMData( const WLEMData& emd ) :
                boost::enable_shared_from_this< WLEMData >()
{
    // C++11 supports "delegating constructors". So default initialization could be moved to default constructor.
    m_data.reset( new DataT() );

    if( !emd.m_chanNames || emd.m_chanNames->empty() )
    {
        wlog::info( "WDataSetEMMEMD" ) << "No channel names available! Channels will be numbered.";
        m_chanNames = WLArrayList< std::string >::instance();
        // Using prefix to avoid ambiguous matchings in MNE library.
        const std::string modName = WLEModality::name( emd.getModalityType() );
        std::stringstream sstream;
        const size_t chanSize = emd.getNrChans();
        for( size_t i = 0; i < chanSize; ++i )
        {
            sstream << modName << std::setw( 3 ) << std::setfill( '0' ) << i + 1;
            m_chanNames->push_back( sstream.str() );
            sstream.str( "" );
            sstream.clear();
        }
    }
    else
    {
        m_chanNames = emd.m_chanNames;
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

    m_badChannels.reset( new ChannelList() );
}

WLEMData::~WLEMData()
{
}

WLEMData::DataT& WLEMData::getData() const
{
    return *m_data;
}

WLEMData::DataSPtr WLEMData::getDataBadChannels() const
{
    WLEMData::DataSPtr dataPtr( new WLEMData::DataT( m_data->rows() - m_badChannels->size(), getSamplesPerChan() ) );
    WLEMData::DataT& data = *dataPtr;

    size_t row = 0, it;
    for( it = 0; it < getNrChans(); ++it )
    {
        if( isBadChannel( it + 1 ) )
        {
            continue;
        }

        data.row( row++ ) = m_data->row( it );
    }

    return dataPtr;
}

void WLEMData::setData( DataSPtr data )
{
    m_data = data;
}

float WLEMData::getAnalogHighPass() const
{
    return m_analogHighPass;
}

float WLEMData::getAnalogLowPass() const
{
    return m_analogLowPass;
}

WLArrayList< std::string >::SPtr WLEMData::getChanNames()
{
    return m_chanNames;
}

WLArrayList< std::string >::ConstSPtr WLEMData::getChanNames() const
{
    return m_chanNames;
}

LaBP::WEUnit::Enum WLEMData::getChanUnit() const
{
    return m_chanUnit;
}

LaBP::WEExponent::Enum WLEMData::getChanUnitExp() const
{
    return m_chanUnitExp;
}

LaBP::WECoordSystemName::Enum WLEMData::getCoordSystem() const
{
    return m_CoordSystem;
}

uint32_t WLEMData::getDataBuffSizePerChan() const
{
    return m_dataBuffSizePerChan;
}

uint32_t WLEMData::getDataOffsetIdx() const
{
    return m_dataOffsetIdx;
}

float WLEMData::getLineFreq()
{
    return m_lineFreq;
}

std::string WLEMData::getMeasurementDeviceName() const
{
    return m_measurementDeviceName;
}

size_t WLEMData::getNrChans() const
{
    if( m_data )
    {
        return m_data->rows();
    }
    return 0;
}

size_t WLEMData::getSamplesPerChan() const
{
    if( m_data )
    {
        return m_data->cols();
    }
    return 0;
}

uint16_t *WLEMData::getOrigIdx() const
{
    return m_origIdx;
}

float WLEMData::getSampFreq() const
{
    return m_sampFreq;
}

float WLEMData::getLength() const
{
    return getSamplesPerChan() / m_sampFreq;
}

WLEMData::ChannelListSPtr WLEMData::getBadChannels() const
{
    return m_badChannels;
}

void WLEMData::setAnalogHighPass( float analogHighPass )
{
    m_analogHighPass = analogHighPass;
}

void WLEMData::setAnalogLowPass( float analogLowPass )
{
    m_analogLowPass = analogLowPass;
}

void WLEMData::setChanNames( WLArrayList< std::string >::SPtr chanNames )
{
    m_chanNames = chanNames;
}

void WLEMData::setChanNames( boost::shared_ptr< std::vector< std::string > > chanNames )
{
    m_chanNames = WLArrayList< std::string >::instance( *chanNames );
}

void WLEMData::setChanUnit( LaBP::WEUnit::Enum chanUnit )
{
    m_chanUnit = chanUnit;
}

void WLEMData::setChanUnitExp( LaBP::WEExponent::Enum chanUnitExp )
{
    m_chanUnitExp = chanUnitExp;
}

void WLEMData::setCoordSystem( LaBP::WECoordSystemName::Enum coordSystem )
{
    m_CoordSystem = coordSystem;
}

void WLEMData::setDataBuffSizePerChan( uint32_t dataBuffSizePerChan )
{
    m_dataBuffSizePerChan = dataBuffSizePerChan;
}

void WLEMData::setDataOffsetIdx( uint32_t dataOffsetIdx )
{
    m_dataOffsetIdx = dataOffsetIdx;
}

void WLEMData::setLineFreq( float lineFreq )
{
    m_lineFreq = lineFreq;
}

void WLEMData::setMeasurementDeviceName( std::string measurementDeviceName )
{
    m_measurementDeviceName = measurementDeviceName;
}

void WLEMData::setOrigIdx( uint16_t *origIdx )
{
    m_origIdx = origIdx;
}

void WLEMData::setSampFreq( float sampFreq )
{
    m_sampFreq = sampFreq;
}

void WLEMData::setBadChannels( WLEMData::ChannelListSPtr badChannels )
{
    m_badChannels = badChannels;
}

std::string WLEMData::channelToString( const ChannelT& data, size_t maxSamples )
{
    const size_t nbSmp = data.size();
    std::stringstream ss;
    for( size_t j = 0; j < maxSamples && j < nbSmp; ++j )
    {
        ss << data( j ) << " ";
    }
    return ss.str();
}

std::string WLEMData::dataToString( const DataT& data, size_t maxChannels, size_t maxSamples )
{
    const size_t nbChan = data.rows();

    std::stringstream ss;
    for( size_t i = 0; i < maxChannels && i < nbChan; ++i )
    {
        ss << "Channel " << i << ": " << channelToString( data.row( i ), maxSamples ) << std::endl;
    }
    return ss.str();
}

bool WLEMData::isBadChannel( size_t channelNo ) const
{
    ChannelList::iterator it;

    for( it = m_badChannels->begin(); it != m_badChannels->end(); ++it )
    {
        if( *it == channelNo )
            return true;
    }

    return false;
}

std::ostream& operator<<( std::ostream &strm, const WLEMData& obj )
{
    strm << WLEMData::CLASS << "::" << WLEModality::name( obj.getModalityType() ) << ": data=" << obj.getNrChans() << "x" << obj.getSamplesPerChan();
    return strm;
}

