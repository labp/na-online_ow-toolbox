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
#include <cstddef>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"
#include "WLEMMEnumTypes.h"
#include "WLEMMSubject.h"

#include "WLEMMeasurement.h"

const std::string WLEMMeasurement::CLASS = "WLEMMeasurement";

WLEMMeasurement::WLEMMeasurement()
{
    m_eventChannels.reset( new EDataT() );
    m_subject.reset( new LaBP::WLEMMSubject() );
    m_profiler.reset( new WLTimeProfiler( CLASS, "lifetime" ) );
}

WLEMMeasurement::WLEMMeasurement( LaBP::WLEMMSubject::SPtr subject )
{
    m_subject = subject;
    m_eventChannels.reset( new EDataT() );
    m_profiler.reset( new WLTimeProfiler( CLASS, "lifetime" ) );
}

WLEMMeasurement::WLEMMeasurement( const WLEMMeasurement& emm )
{
    m_eventChannels.reset( new EDataT() );

    m_profiler.reset( new WLTimeProfiler( *emm.m_profiler ) );

    m_expDescription = emm.m_expDescription;
    m_experimenter = emm.m_experimenter;
    m_subject = emm.m_subject;
}

WLEMMeasurement::~WLEMMeasurement()
{
}

WLEMMeasurement::SPtr WLEMMeasurement::clone() const
{
    WLEMMeasurement::SPtr emm( new WLEMMeasurement( *this ) );
    return emm;
}

WLEMMeasurement::SPtr WLEMMeasurement::newModalityData( WLEMData::SPtr modality )
{
    WLEMMeasurement::SPtr emm( new WLEMMeasurement( *this ) );

    std::vector< WLEMData::SPtr > *newModalityList = new std::vector< WLEMData::SPtr >();

    for( uint i = 0; i < m_modalityList.size(); i++ )
    {
        if( ( m_modalityList[i] )->getModalityType() == modality->getModalityType() )
        {
            newModalityList->push_back( modality );
        }
        else
        {
            newModalityList->push_back( WLEMData::SPtr( m_modalityList[i] ) );
        }
    }
    emm->setModalityList( *newModalityList );

    return emm;
}

void WLEMMeasurement::addModality( boost::shared_ptr< WLEMData > modality )
{
    m_modalityList.push_back( modality );
}

// -----------getter and setter-----------------------------------------------------------------------------

std::string WLEMMeasurement::getExperimenter() const
{
    return m_experimenter;
}

std::string WLEMMeasurement::getExpDescription() const
{
    return m_expDescription;
}

LaBP::WLEMMSubject::SPtr WLEMMeasurement::getSubject()
{
    return m_subject;
}

LaBP::WLEMMSubject::ConstSPtr WLEMMeasurement::getSubject() const
{
    return m_subject;
}

std::vector< WLEMData::SPtr > WLEMMeasurement::getModalityList()
{
    return m_modalityList;
}

void WLEMMeasurement::setModalityList( std::vector< WLEMData::SPtr > list )
{
    m_modalityList.clear();
    m_modalityList = list;
}

size_t WLEMMeasurement::getModalityCount() const
{
    return m_modalityList.size();
}

WLEMData::SPtr WLEMMeasurement::getModality( size_t i )
{
    if( m_modalityList.size() )
    {
        return m_modalityList.at( i );
    }
    else
    {
        throw "Index out of range!";
    }
}

WLEMData::ConstSPtr WLEMMeasurement::getModality( size_t i ) const
{
    if( m_modalityList.size() )
    {
        return m_modalityList.at( i );
    }
    else
    {
        throw "Index out of range!";
    }
}

WLEMData::SPtr WLEMMeasurement::getModality( LaBP::WEModalityType::Enum type )
{
    for( std::vector< WLEMData::SPtr >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        if( m_modalityList.at( i )->getModalityType() == type )
        {
            return m_modalityList.at( i );
        }
    }
    throw "Modality type not available!";
}

WLEMData::ConstSPtr WLEMMeasurement::getModality( LaBP::WEModalityType::Enum type ) const
{
    for( std::vector< WLEMData::SPtr >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        if( m_modalityList.at( i )->getModalityType() == type )
        {
            return m_modalityList.at( i );
        }
    }
    throw "Modality type not available!";
}

std::set< LaBP::WEModalityType::Enum > WLEMMeasurement::getModalityTypes() const
{
    std::set< LaBP::WEModalityType::Enum > enums;
    for( std::vector< WLEMData::SPtr >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        enums.insert( m_modalityList.at( i )->getModalityType() );
    }
    return enums;
}

bool WLEMMeasurement::hasModality( LaBP::WEModalityType::Enum type ) const
{
    for( std::vector< boost::shared_ptr< WLEMData > >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        if( m_modalityList.at( i )->getModalityType() == type )
        {
            return true;
        }
    }
    return false;
}

void WLEMMeasurement::setExperimenter( std::string experimenter )
{
    m_experimenter = experimenter;
}

void WLEMMeasurement::setExpDescription( std::string expDescription )
{
    m_expDescription = expDescription;
}

void WLEMMeasurement::setSubject( LaBP::WLEMMSubject::SPtr subject )
{
    m_subject = subject;
}

boost::shared_ptr< WLEMMeasurement::EDataT > WLEMMeasurement::getEventChannels() const
{
    return m_eventChannels;
}

void WLEMMeasurement::setEventChannels( boost::shared_ptr< EDataT > data )
{
    m_eventChannels = data;
}

void WLEMMeasurement::addEventChannel( EChannelT& data )
{
    m_eventChannels->push_back( data );
}

WLEMMeasurement::EChannelT& WLEMMeasurement::getEventChannel( int i ) const
{
    return m_eventChannels->at( i );
}

size_t WLEMMeasurement::getEventChannelCount() const
{
    return m_eventChannels->size();
}

WLTimeProfiler::SPtr WLEMMeasurement::getTimeProfiler()
{
    return m_profiler;
}

WLTimeProfiler::ConstSPtr WLEMMeasurement::getTimeProfiler() const
{
    return m_profiler;
}

void WLEMMeasurement::setTimeProfiler( WLTimeProfiler::SPtr profiler )
{
    m_profiler = profiler;
}

WLTimeProfiler::SPtr WLEMMeasurement::createAndAddProfiler( std::string clazz, std::string action )
{
    WLTimeProfiler::SPtr profiler( new WLTimeProfiler( clazz, action ) );
    m_profiler->addChild( profiler );
    return profiler;
}
