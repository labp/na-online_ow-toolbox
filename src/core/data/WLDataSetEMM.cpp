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

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/dataHandler/WDataSet.h>

#include "core/dataHandler/WDataSetEMMEEG.h"
#include "core/dataHandler/WDataSetEMMEMD.h"
#include "core/dataHandler/WDataSetEMMEnumTypes.h"
#include "core/dataHandler/WDataSetEMMSubject.h"

#include "WLDataSetEMM.h"

// prototype instance as singleton
boost::shared_ptr< WPrototyped > LaBP::WLDataSetEMM::m_prototype = boost::shared_ptr< WPrototyped >();

LaBP::WLDataSetEMM::WLDataSetEMM() :
                WDataSet()
{
    m_eventChannels.reset( new std::vector< std::vector< int > >() );
    m_subject.reset( new LaBP::WDataSetEMMSubject() );
    m_profiler.reset( new LaBP::WLTimeProfiler( getName(), "lifetime" ) );
}

LaBP::WLDataSetEMM::WLDataSetEMM( boost::shared_ptr< LaBP::WDataSetEMMSubject > subject ) :
                WDataSet()
{
    m_subject = subject;
    m_eventChannels.reset( new std::vector< std::vector< int > >() );
    m_profiler.reset( new LaBP::WLTimeProfiler( getName(), "lifetime" ) );
}

LaBP::WLDataSetEMM::WLDataSetEMM( WPropBool dataUpdate ) :
                WDataSet()
{
    m_dataUpdate = dataUpdate;

    m_eventChannels.reset( new std::vector< std::vector< int > >() );
    m_profiler.reset( new LaBP::WLTimeProfiler( getName(), "lifetime" ) );
}

LaBP::WLDataSetEMM::WLDataSetEMM( const LaBP::WLDataSetEMM& emm ) :
                WDataSet()
{
    m_eventChannels.reset( new std::vector< std::vector< int > >() );

    m_profiler.reset( new WLTimeProfiler( *emm.m_profiler ) );

    m_dataUpdate = emm.m_dataUpdate;
    m_expDescription = emm.m_expDescription;
    m_experimenter = emm.m_experimenter;
    m_subject = emm.m_subject;
}

LaBP::WLDataSetEMM::~WLDataSetEMM()
{

}

LaBP::WLDataSetEMM::SPtr LaBP::WLDataSetEMM::clone() const
{
    LaBP::WLDataSetEMM::SPtr emm( new WLDataSetEMM( *this ) );
    return emm;
}

const std::string LaBP::WLDataSetEMM::getName() const
{
    return "WDataSetEMM";
}

const std::string LaBP::WLDataSetEMM::getDescription() const
{
    return "Contains WDataSetEMM data.";
}

boost::shared_ptr< WPrototyped > LaBP::WLDataSetEMM::getPrototype()
{
    if( !m_prototype )
    {
        m_prototype = boost::shared_ptr< WPrototyped >( new WLDataSetEMM() );
    }

    return m_prototype;
}

boost::shared_ptr< LaBP::WLDataSetEMM > LaBP::WLDataSetEMM::newModalityData( boost::shared_ptr< LaBP::WDataSetEMMEMD > modality )
{
    boost::shared_ptr< LaBP::WLDataSetEMM > emm = boost::shared_ptr< LaBP::WLDataSetEMM >( new LaBP::WLDataSetEMM( *this ) );

    //    for (std::vector< boost::shared_ptr< LaBP::WDataSetEMMEMD > >::iterator it = emm->getModalityList().begin(); it != emm->getModalityList().end(); ++it)
    //    {
    //    	LaBP::WModalityType m1 = (**it).getModality();
    //    	LaBP::WModalityType m2 = modality->getModality();
    //        if ((*it)->getModality() == modality->getModality())
    //        {
    //            (*it) = modality;
    //        }
    //    }

    //    for (int i = 0; i < emm->getModalityList().size(); i++) {
    //    	if ((emm->getModalityList()[i])->getModality() == modality->getModality()){
    //    		(emm->getModalityList()[i]) = modality;
    //    	}
    //	}

    std::vector< boost::shared_ptr< LaBP::WDataSetEMMEMD > > *newModalityList = new std::vector<
                    boost::shared_ptr< LaBP::WDataSetEMMEMD > >();

    for( uint i = 0; i < m_modalityList.size(); i++ )
    {
        if( ( m_modalityList[i] )->getModalityType() == modality->getModalityType() )
        {
            newModalityList->push_back( modality );
        }
        else
        {
            newModalityList->push_back( boost::shared_ptr< LaBP::WDataSetEMMEMD >( m_modalityList[i] ) );
        }
    }
    emm->setModalityList( *newModalityList );

    return emm;
}

void LaBP::WLDataSetEMM::addModality( boost::shared_ptr< LaBP::WDataSetEMMEMD > modality )
{
    m_modalityList.push_back( modality );
}

void LaBP::WLDataSetEMM::fireDataUpdateEvent()
{
    // TODO(kaehler): Check if this works right
    m_dataUpdate->set( true, false );
    m_dataUpdate->set( false, false );
}

//boost::shared_ptr< LaBP::WDataSetEMM > LaBP::WDataSetEMM::dataChanged(boost::shared_ptr< std::vector< std::vector< double > > > newData)
//{getModality
//    LaBP::WDataSetEMM emm(*this);
//    boost::shared_ptr< std::vector< boost::shared_ptr< LaBP::WDataSetEMMEMD > > > modalityList(new std::vector< boost::shared_ptr< LaBP::WDataSetEMMEMD > >());
//    emm.setModalityList(modalityList);
//
//    boost::shared_ptr< LaBP::WDataSetEMM > returnEmm = boost::shared_ptr< LaBP::WDataSetEMM >(emm);
//    return returnEmm;
//}

// -----------getter and setter-----------------------------------------------------------------------------

std::string LaBP::WLDataSetEMM::getExperimenter() const
{
    return m_experimenter;
}

std::string LaBP::WLDataSetEMM::getExpDescription() const
{
    return m_expDescription;
}

LaBP::WDataSetEMMSubject::SPtr LaBP::WLDataSetEMM::getSubject()
{
    return m_subject;
}

LaBP::WDataSetEMMSubject::ConstSPtr LaBP::WLDataSetEMM::getSubject() const
{
    return m_subject;
}

std::vector< LaBP::WDataSetEMMEMD::SPtr > LaBP::WLDataSetEMM::getModalityList()
{
    return m_modalityList;
}

void LaBP::WLDataSetEMM::setModalityList( std::vector< LaBP::WDataSetEMMEMD::SPtr > list )
{
    m_modalityList.clear();
    m_modalityList = list;
}

size_t LaBP::WLDataSetEMM::getModalityCount() const
{
    return m_modalityList.size();
}

LaBP::WDataSetEMMEMD::SPtr LaBP::WLDataSetEMM::getModality( size_t i )
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

LaBP::WDataSetEMMEMD::ConstSPtr LaBP::WLDataSetEMM::getModality( size_t i ) const
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

LaBP::WDataSetEMMEMD::SPtr LaBP::WLDataSetEMM::getModality( LaBP::WEModalityType::Enum type )
{
    for( std::vector< boost::shared_ptr< WDataSetEMMEMD > >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        if( m_modalityList.at( i )->getModalityType() == type )
        {
            return m_modalityList.at( i );
        }
    }
    throw "Modality type not available!";
}

LaBP::WDataSetEMMEMD::ConstSPtr LaBP::WLDataSetEMM::getModality( LaBP::WEModalityType::Enum type ) const
{
    for( std::vector< boost::shared_ptr< WDataSetEMMEMD > >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        if( m_modalityList.at( i )->getModalityType() == type )
        {
            return m_modalityList.at( i );
        }
    }
    throw "Modality type not available!";
}

std::set< LaBP::WEModalityType::Enum > LaBP::WLDataSetEMM::getModalityTypes() const
{
    std::set< LaBP::WEModalityType::Enum > enums;
    for( std::vector< boost::shared_ptr< WDataSetEMMEMD > >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        enums.insert( m_modalityList.at( i )->getModalityType() );
    }
    return enums;
}

bool LaBP::WLDataSetEMM::hasModality( LaBP::WEModalityType::Enum type ) const
{
    for( std::vector< boost::shared_ptr< WDataSetEMMEMD > >::size_type i = 0; i < m_modalityList.size(); ++i )
    {
        if( m_modalityList.at( i )->getModalityType() == type )
        {
            return true;
        }
    }
    return false;
}

void LaBP::WLDataSetEMM::setExperimenter( std::string experimenter )
{
    m_experimenter = experimenter;
}

void LaBP::WLDataSetEMM::setExpDescription( std::string expDescription )
{
    m_expDescription = expDescription;
}

void LaBP::WLDataSetEMM::setSubject( boost::shared_ptr< WDataSetEMMSubject > subject )
{
    m_subject = subject;
}

boost::shared_ptr< std::vector< std::vector< int > > > LaBP::WLDataSetEMM::getEventChannels() const
{
    return m_eventChannels;
}

void LaBP::WLDataSetEMM::setEventChannels( boost::shared_ptr< EDataT > data )
{
    m_eventChannels = data;
}

void LaBP::WLDataSetEMM::addEventChannel( EChannelT& data )
{
    m_eventChannels->push_back( data );
}

LaBP::WLDataSetEMM::EChannelT& LaBP::WLDataSetEMM::getEventChannel( int i ) const
{
    return m_eventChannels->at( i );
}

size_t LaBP::WLDataSetEMM::getEventChannelCount() const
{
    return m_eventChannels->size();
}

LaBP::WLTimeProfiler::SPtr LaBP::WLDataSetEMM::getTimeProfiler()
{
    return m_profiler;
}

LaBP::WLTimeProfiler::ConstSPtr LaBP::WLDataSetEMM::getTimeProfiler() const
{
    return m_profiler;
}

void LaBP::WLDataSetEMM::setTimeProfiler( LaBP::WLTimeProfiler::SPtr profiler )
{
    m_profiler = profiler;
}

LaBP::WLTimeProfiler::SPtr LaBP::WLDataSetEMM::createAndAddProfiler( std::string clazz, std::string action )
{
    WLTimeProfiler::SPtr profiler( new WLTimeProfiler( clazz, action ) );
    m_profiler->addChild( profiler );
    return profiler;
}

