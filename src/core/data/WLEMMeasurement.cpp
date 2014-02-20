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
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/common/exceptions/WNotFound.h>

#include "core/data/enum/WLEModality.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/emd/WLEMData.h"

#include "WLEMMeasurement.h"

const std::string WLEMMeasurement::CLASS = "WLEMMeasurement";

WLEMMeasurement::WLEMMeasurement()
{
    m_transDevToFid.setIdentity();
    m_transFidToACPC.setIdentity();
    m_eventChannels.reset( new EDataT() );
    m_subject.reset( new WLEMMSubject() );
    m_profiler.reset( new WLLifetimeProfiler( CLASS, "lifetime" ) );

    m_digPoints = WLList< WLDigPoint >::instance();
}

WLEMMeasurement::WLEMMeasurement( WLEMMSubject::SPtr subject )
{
    m_transDevToFid.setIdentity();
    m_transFidToACPC.setIdentity();
    m_subject = subject;
    m_eventChannels.reset( new EDataT() );
    m_profiler.reset( new WLLifetimeProfiler( CLASS, "lifetime" ) );
    m_digPoints = WLList< WLDigPoint >::instance();
}

WLEMMeasurement::WLEMMeasurement( const WLEMMeasurement& emm )
{
    m_eventChannels.reset( new EDataT() );

    m_profiler.reset( new WLLifetimeProfiler( *emm.m_profiler ) );

    m_expDescription = emm.m_expDescription;
    m_experimenter = emm.m_experimenter;
    m_subject = emm.m_subject;
    m_digPoints = emm.m_digPoints;
    m_transDevToFid = emm.m_transDevToFid;
    m_transFidToACPC = emm.m_transFidToACPC;
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

bool WLEMMeasurement::addModality( WLEMData::SPtr modality )
{
    WLEModality::Enum m = modality->getModalityType();
    if( !WLEModality::isMEGCoil( m ) )
    {
        m_modalityList.push_back( modality );
        return true;
    }
    else
    {
        wlog::warn( CLASS ) << "Skipping modality: " << WLEModality::name( m );
        return false;
    }

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

WLEMMSubject::SPtr WLEMMeasurement::getSubject()
{
    return m_subject;
}

WLEMMSubject::ConstSPtr WLEMMeasurement::getSubject() const
{
    return m_subject;
}

std::vector< WLEMData::SPtr > WLEMMeasurement::getModalityList()
{
    return m_modalityList;
}

size_t WLEMMeasurement::setModalityList( const std::vector< WLEMData::SPtr >&list )
{
    m_modalityList.clear();
    std::vector< WLEMData::SPtr >::const_iterator it;
    for( it = list.begin(); it != list.end(); ++it )
    {
        const WLEModality::Enum m = ( *it )->getModalityType();
        if( !WLEModality::isMEGCoil( m ) )
        {
            m_modalityList.push_back( ( *it ) );
        }
        else
        {
            wlog::warn( CLASS ) << "Skipping modality: " << WLEModality::name( m );
        }
    }
    return m_modalityList.size();
}

size_t WLEMMeasurement::getModalityCount() const
{
    return m_modalityList.size();
}

WLEMData::SPtr WLEMMeasurement::getModality( size_t i )
{
    // .at() throws an std::out_of_range
    return m_modalityList.at( i );
}

WLEMData::ConstSPtr WLEMMeasurement::getModality( size_t i ) const
{
    // .at() throws an std::out_of_range
    return m_modalityList.at( i );
}

WLEMData::SPtr WLEMMeasurement::getModality( WLEModality::Enum type )
{
    for( std::vector< WLEMData::SPtr >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        if( m_modalityList.at( i )->getModalityType() == type )
        {
            return m_modalityList.at( i );
        }
    }
    throw WNotFound( "Modality type not available!" );
}

WLEMData::ConstSPtr WLEMMeasurement::getModality( WLEModality::Enum type ) const
{
    for( std::vector< WLEMData::SPtr >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        if( m_modalityList.at( i )->getModalityType() == type )
        {
            return m_modalityList.at( i );
        }
    }
    throw WNotFound( "Modality type not available!" );
}

std::set< WLEModality::Enum > WLEMMeasurement::getModalityTypes() const
{
    std::set< WLEModality::Enum > enums;
    for( std::vector< WLEMData::SPtr >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        enums.insert( m_modalityList.at( i )->getModalityType() );
    }
    return enums;
}

bool WLEMMeasurement::hasModality( WLEModality::Enum type ) const
{
    for( std::vector< WLEMData::SPtr >::size_type i = 0; i < m_modalityList.size(); ++i )
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

void WLEMMeasurement::setSubject( WLEMMSubject::SPtr subject )
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

WLEMMeasurement::EChannelT& WLEMMeasurement::getEventChannel( WLChanIdxT i ) const
{
    return m_eventChannels->at( i );
}

WLChanNrT WLEMMeasurement::getEventChannelCount() const
{
    return m_eventChannels->size();
}

WLLifetimeProfiler::SPtr WLEMMeasurement::getProfiler()
{
    return m_profiler;
}

WLLifetimeProfiler::ConstSPtr WLEMMeasurement::getProfiler() const
{
    return m_profiler;
}

void WLEMMeasurement::setProfiler( WLLifetimeProfiler::SPtr profiler )
{
    m_profiler = profiler;
}

WLList< WLDigPoint >::SPtr WLEMMeasurement::getDigPoints()
{
    return m_digPoints;
}

WLList< WLDigPoint >::ConstSPtr WLEMMeasurement::getDigPoints() const
{
    return m_digPoints;
}

void WLEMMeasurement::setDigPoints( WLList< WLDigPoint >::SPtr digPoints )
{
    m_digPoints = digPoints;
}

WLList< WLDigPoint >::SPtr WLEMMeasurement::getDigPoints( WLEPointType::Enum kind ) const
{
    WLList< WLDigPoint >::SPtr digForKind( new WLList< WLDigPoint >() );
    WLList< WLDigPoint >::const_iterator cit;
    for( cit = m_digPoints->begin(); cit != m_digPoints->end(); ++cit )
    {
        if( cit->getKind() == kind )
        {
            digForKind->push_back( *cit );
        }
    }
    return digForKind;
}

const WLMatrix4::Matrix4T& WLEMMeasurement::getDevToFidTransformation() const
{
    return m_transDevToFid;
}

void WLEMMeasurement::setDevToFidTransformation( const WLMatrix4::Matrix4T& mat )
{
    m_transDevToFid = mat;
}

const WLMatrix4::Matrix4T& WLEMMeasurement::getFidToACPCTransformation() const
{
    return m_transFidToACPC;
}

void WLEMMeasurement::setFidToACPCTransformation( const WLMatrix4::Matrix4T& mat )
{
    m_transFidToACPC = mat;
}
