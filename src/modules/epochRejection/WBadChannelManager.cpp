/*
 * WBadChannelManager.cpp
 *
 *  Created on: 08.11.2013
 *      Author: maschke
 */

#include "WBadChannelManager.h"

WBadChannelManager::WBadChannelManager()
{

}

WBadChannelManager::~WBadChannelManager()
{

}

WBadChannelManager *WBadChannelManager::instance()
{
    static WBadChannelManager _instance;

    return &_instance;
}

void WBadChannelManager::addElement( const LaBP::WEModalityType::Enum& mod, const size_t& channel )
{
    if( m_map.find( mod ) == m_map.end() )
    {
        m_map.insert( std::pair< LaBP::WEModalityType::Enum, WGenericList< size_t > >( mod, WGenericList< size_t >() ) );
    }

    m_map.find( mod )->second.addElement( channel );
}

void WBadChannelManager::removeAt( LaBP::WEModalityType::Enum const& mod, size_t const& channel )
{
    if( IsEmpty() )
    {
        return;
    }

    if( m_map.count( mod ) > 0 )
    {
        m_map.find( mod )->second.removeAt( channel );
    }
}
