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

#include <algorithm>
#include <sstream>
#include <string>
#include <ostream> // std::endl

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>

#include "WLEMData.h"

const std::string WLEMData::CLASS = "WLEMData";

const WLFreqT WLEMData::UNDEFINED_FREQ = -1;

WLEMData::WLEMData() :
                boost::enable_shared_from_this< WLEMData >()
{
    m_data.reset( new DataT() );
    m_chanNames = WLArrayList< std::string >::instance();
    m_chanUnit = WLEUnit::NONE;
    m_chanUnitExp = WLEExponent::UNKNOWN;
    m_sampFreq = UNDEFINED_FREQ;
    m_lineFreq = UNDEFINED_FREQ;
    m_analogHighPass = UNDEFINED_FREQ;
    m_analogLowPass = UNDEFINED_FREQ;
    m_CoordSystem = WLECoordSystem::UNKNOWN;

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
    m_lineFreq = emd.getLineFreq();
    m_chanUnit = emd.getChanUnit();
    m_chanUnitExp = emd.getChanUnitExp();
    m_analogHighPass = emd.getAnalogHighPass();
    m_analogLowPass = emd.getAnalogLowPass();
    m_CoordSystem = emd.getCoordSystem();

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
        if( isBadChannel( it ) )
        {
            continue;
        }

        data.row( row++ ) = m_data->row( it );
    }

    return dataPtr;
}

WLEMData::DataSPtr WLEMData::getDataBadChannels( ChannelListSPtr badChans ) const
{
    if(badChans == 0)
    {
        return getDataBadChannels();
    }

    WLEMData::DataSPtr dataPtr( new WLEMData::DataT( m_data->rows() - m_badChannels->size(), getSamplesPerChan() ) );
    WLEMData::DataT& data = *dataPtr;

    size_t row = 0, it;
    for( it = 0; it < getNrChans(); ++it )
    {
        if( isBadChannel( it ) || std::find(badChans->begin(), badChans->end(), it) != badChans->end() )
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

WLFreqT WLEMData::getAnalogHighPass() const
{
    return m_analogHighPass;
}

WLFreqT WLEMData::getAnalogLowPass() const
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

WLEUnit::Enum WLEMData::getChanUnit() const
{
    return m_chanUnit;
}

WLEExponent::Enum WLEMData::getChanUnitExp() const
{
    return m_chanUnitExp;
}

WLECoordSystem::Enum WLEMData::getCoordSystem() const
{
    return m_CoordSystem;
}

WLFreqT WLEMData::getLineFreq() const
{
    return m_lineFreq;
}

std::string WLEMData::getMeasurementDeviceName() const
{
    return m_measurementDeviceName;
}

WLChanNrT WLEMData::getNrChans() const
{
    if( m_data )
    {
        return m_data->rows();
    }
    return 0;
}

WLSampleNrT WLEMData::getSamplesPerChan() const
{
    if( m_data )
    {
        return m_data->cols();
    }
    return 0;
}

WLFreqT WLEMData::getSampFreq() const
{
    return m_sampFreq;
}

WLTimeT WLEMData::getLength() const
{
    return getSamplesPerChan() / m_sampFreq;
}

WLEMData::ChannelListSPtr WLEMData::getBadChannels() const
{
    return m_badChannels;
}

void WLEMData::setAnalogHighPass( WLFreqT analogHighPass )
{
    m_analogHighPass = analogHighPass;
}

void WLEMData::setAnalogLowPass( WLFreqT analogLowPass )
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

void WLEMData::setChanUnit( WLEUnit::Enum chanUnit )
{
    m_chanUnit = chanUnit;
}

void WLEMData::setChanUnitExp( WLEExponent::Enum chanUnitExp )
{
    m_chanUnitExp = chanUnitExp;
}

void WLEMData::setCoordSystem( WLECoordSystem::Enum coordSystem )
{
    m_CoordSystem = coordSystem;
}

void WLEMData::setLineFreq( float lineFreq )
{
    m_lineFreq = lineFreq;
}

void WLEMData::setMeasurementDeviceName( std::string measurementDeviceName )
{
    m_measurementDeviceName = measurementDeviceName;
}

void WLEMData::setSampFreq( WLFreqT sampFreq )
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
    return !( std::find( m_badChannels->begin(), m_badChannels->end(), channelNo ) == m_badChannels->end() );
}
