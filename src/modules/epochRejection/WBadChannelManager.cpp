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

#include "WBadChannelManager.h"

WBadChannelManager::WBadChannelManager()
{
    m_map.reset( new ChannelMap() );
}

WBadChannelManager::~WBadChannelManager()
{

}

WBadChannelManager *WBadChannelManager::instance()
{
    static WBadChannelManager _instance;

    return &_instance;
}

void WBadChannelManager::addChannel( const LaBP::WEModalityType::Enum& mod, const size_t& channel )
{
    if( m_map->find( mod ) == m_map->end() )
    {
        m_map->insert(
                        std::pair< LaBP::WEModalityType::Enum, WLEMData::ChannelListSPtr >( mod,
                                        WLEMData::ChannelListSPtr( new WLEMData::ChannelList ) ) );
    }
    m_map->find( mod )->second->push_back( channel );
}

void WBadChannelManager::removeChannel( LaBP::WEModalityType::Enum const& mod, size_t const& channel )
{
    if( m_map->size() == 0 )
    {
        return;
    }

    if( m_map->count( mod ) > 0 )
    {
        m_map->find( mod )->second->remove( channel );
    }
}

bool WBadChannelManager::isMapEmpty() const
{
    return m_map->size() == 0;
}

bool WBadChannelManager::isChannelBad( const LaBP::WEModalityType::Enum& modality, const size_t channelNo ) const
{
    if(m_map->count(modality) == 0)
        return false;

    WLEMData::ChannelListSPtr list = m_map->find(modality)->second;
    WLEMData::ChannelList::iterator it;

    for(it = list->begin(); it != list->end(); ++it)
    {
        if(*it == channelNo)
            return true;
    }

    return false;
}

bool WBadChannelManager::hasBadChannels( const LaBP::WEModalityType::Enum& modality ) const
{
    return m_map->count(modality) > 0;
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

size_t WBadChannelManager::countChannels( const LaBP::WEModalityType::Enum& modality ) const
{
    if( m_map->count( modality ) == 0 )
        return 0;

    return m_map->find( modality )->second->size();
}

WLEMData::ChannelListSPtr WBadChannelManager::getChannelList( const LaBP::WEModalityType::Enum& mod )
{
    if( m_map->count( mod ) > 0 )
    {
        return m_map->find( mod )->second;
    }

    return WLEMData::ChannelListSPtr();
}
