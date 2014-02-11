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

#include "core/data/WLEMMeasurement.h"

#include "WBadEpochManager.h"

WBadEpochManager* WBadEpochManager::m_instance = 0;

WBadEpochManager::WBadEpochManager()
{
    m_buffer.reset( new boost::circular_buffer< WLEMMeasurement::SPtr >( 5 ) );
}

WBadEpochManager::~WBadEpochManager()
{

}

WBadEpochManager *WBadEpochManager::instance()
{
    if( m_instance == 0 )
        m_instance = new WBadEpochManager();

    return m_instance;
}

WBadEpochManager::CircBuffSPtr WBadEpochManager::getBuffer()
{
    return m_buffer;
}

void WBadEpochManager::setBuffer( WBadEpochManager::CircBuffSPtr buffer )
{
    m_buffer = buffer;
}
