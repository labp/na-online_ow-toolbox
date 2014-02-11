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

#include <map>

#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"

#include "WBadChannelManager.h"

WBadChannelManager* WBadChannelManager::m_instance = 0;

WBadChannelManager::WBadChannelManager()
{
    m_map.reset( new ChannelMap() );
}

WBadChannelManager::~WBadChannelManager()
{

}

WBadChannelManager *WBadChannelManager::instance()
{
    if( m_instance == 0 )
        m_instance = new WBadChannelManager();

    return m_instance;
}

void WBadChannelManager::addChannel( const WLEModality::Enum& mod, const size_t& channel )
{
    if( m_map->count( mod ) == 0 )
    {
        m_map->insert(
                        std::pair< WLEModality::Enum, WLEMData::ChannelListSPtr >( mod,
                                        WLEMData::ChannelListSPtr( new WLEMData::ChannelList ) ) );
    }
    else
    {
        if( std::find( m_map->at( mod )->begin(), m_map->at( mod )->end(), channel ) != m_map->at( mod )->end() )
            return;
    }

    m_map->at( mod )->push_back( channel );
}

void WBadChannelManager::removeChannel( WLEModality::Enum const& mod, size_t const& channel )
{
    if( m_map->size() == 0 )
    {
        return;
    }

    if( m_map->count( mod ) > 0 )
    {
        m_map->at( mod )->remove( channel );

        if( m_map->at( mod )->empty() )
            m_map->erase( mod );
    }
}

bool WBadChannelManager::isMapEmpty() const
{
    return m_map->size() == 0;
}

bool WBadChannelManager::isChannelBad( const WLEModality::Enum& modality, const size_t channelNo ) const
{
    if( m_map->count( modality ) == 0 )
        return false;

    WLEMData::ChannelListSPtr list = m_map->find( modality )->second;
    WLEMData::ChannelList::iterator it;

    for( it = list->begin(); it != list->end(); ++it )
    {
        if( *it == channelNo )
            return true;
    }

    return false;
}

bool WBadChannelManager::hasBadChannels( const WLEModality::Enum& modality ) const
{
    if(m_map->empty())
        return false;

    if(m_map->count(modality) == 0)
        return false;

    if(m_map->at(modality)->empty())
            return false;

    return true;
}

size_t WBadChannelManager::countChannels() const
{
    size_t count = 0;

    ChannelMap::iterator it;

    for( it = m_map->begin(); it != m_map->end(); ++it )
    {
        count += it->second->size();
    }

    return count;
}

size_t WBadChannelManager::countChannels( const WLEModality::Enum& modality ) const
{
    if( m_map->count( modality ) == 0 )
        return 0;

    return m_map->find( modality )->second->size();
}

WLEMData::ChannelListSPtr WBadChannelManager::getChannelList( const WLEModality::Enum& mod )
{
    if( m_map->count( mod ) == 0 )
        return WLEMData::ChannelListSPtr();
    else
        if( m_map->at( mod )->empty() )
            return WLEMData::ChannelListSPtr();

    WLEMData::ChannelListSPtr listSPtr( new WLEMData::ChannelList() );
    WLEMData::ChannelList& list = *listSPtr;

    WLEMData::ChannelList::iterator it;
    for( it = m_map->at( mod )->begin(); it != m_map->at( mod )->end(); ++it )
    {
        list.push_back( *it );
    }

    return listSPtr;
}

WBadChannelManager::ChannelMap_SPtr WBadChannelManager::getChannelMap()
{
    if(m_map->empty())
        return WBadChannelManager::ChannelMap_SPtr();

    WBadChannelManager::ChannelMap_SPtr mapSPtr( new WBadChannelManager::ChannelMap() );
    WBadChannelManager::ChannelMap& map = *mapSPtr;
    WBadChannelManager::ChannelMap::iterator it;
    for(it = m_map->begin(); it != m_map->end(); ++it)
    {
        if(hasBadChannels(it->first))
        {
            map.insert(WBadChannelManager::ChannelMap::value_type(it->first, getChannelList(it->first)));
        }
    }

    return mapSPtr;
}

void WBadChannelManager::merge( WBadChannelManager::ChannelMap_SPtr mapToMerge )
{
    ChannelMap::iterator it;

    for( it = mapToMerge->begin(); it != mapToMerge->end(); ++it )
    {
        if( !mapToMerge->at( it->first )->empty() )
            mapToMerge->at( it->first )->sort();

        if( m_map->count( it->first ) )
        {
            if( !m_map->at( it->first )->empty() )
                m_map->at( it->first )->sort();

            m_map->at( it->first )->merge( *mapToMerge->at( it->first ) );
            m_map->at( it->first )->unique(); // remove channel duplicates
        }
        else
        {
            // copy channels to a new list pointer
            WLEMData::ChannelListSPtr newList( new WLEMData::ChannelList( *mapToMerge->at( it->first ) ) );

            m_map->insert( ChannelMap::value_type( it->first, newList ) );
        }
    }
}

void WBadChannelManager::merge( const WLEModality::Enum& mod, WLEMData::ChannelListSPtr channels )
{
    if( !channels->empty() )
        channels->sort();

    if( m_map->count( mod ) )
    {
        if( !m_map->at( mod )->empty() )
            m_map->at( mod )->sort();

        m_map->at( mod )->merge( *channels );
        m_map->at( mod )->unique(); // remove channel duplicates
    }
    else
    {
        // copy channels to a new list pointer
        WLEMData::ChannelListSPtr newList( new WLEMData::ChannelList( *channels ) );

        m_map->insert( ChannelMap::value_type( mod, newList ) );
    }
}

void WBadChannelManager::reset()
{
    m_map->clear();
}

