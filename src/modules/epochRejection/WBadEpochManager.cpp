//---------------------------------------------------------------------------
//
// Project: NA-Online ( http://www.labp.htwk-leipzig.de )
//
// Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
//
// This file is part of NA-Online.
//
// NA-Online is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NA-Online is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
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

size_t WBadEpochManager::getBufferSize()
{
    return m_buffer->capacity();
}

void WBadEpochManager::setBuffer( WBadEpochManager::CircBuffSPtr buffer )
{
    m_buffer = buffer;
}

void WBadEpochManager::resizeBuffer( size_t size )
{
    m_buffer->set_capacity( size );
}

void WBadEpochManager::reset()
{
    m_buffer->clear();
}
