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

#include <algorithm>
#include <string>

#include <modules/ftRtClient/ftbClient/container/WLSmartStorage.h>

WLSmartStorage::WLSmartStorage()
{
    init();
}

WLSmartStorage::WLSmartStorage( std::string value )
{
    setData( value.c_str(), value.length() );
}

WLSmartStorage::WLSmartStorage( const void* data, size_t size )
{
    setData( ( const char* )data, size );
}

std::string WLSmartStorage::toString() const
{
    return std::string( m_data->data(), m_data->size() );
}

const void *WLSmartStorage::getData() const
{
    return m_data->data();
}

void *WLSmartStorage::getData()
{
    return m_data->data();
}

size_t WLSmartStorage::getSize() const
{
    return m_data->size();
}

void WLSmartStorage::clear()
{
    init();
}

void WLSmartStorage::setData( const void * data, size_t size )
{
    init();

    m_data->resize( size );

    const char *ptr = ( const char * )data;

    std::copy( ptr, ptr + size, m_data->data() );
}

void WLSmartStorage::setData( std::string value )
{
    setData( value.c_str(), value.length() );
}

void WLSmartStorage::append( const void * data, size_t size )
{
    const char *src = ( const char* )data;

    std::copy( src, src + size, std::back_inserter( *m_data ) );
}

void WLSmartStorage::append( std::string value )
{
    append( value.c_str(), value.length() );
}

void WLSmartStorage::init()
{
    m_data.reset( new ContainerT );
}
