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

#include <boost/foreach.hpp>

#include "WFTChannelNames.h"

WFTChannelNames::WFTChannelNames()
{
    m_namesMap.reset( new ChanNamesMapT );

    m_chunkdef.type = WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES;
}

WLArrayList< std::string >::SPtr WFTChannelNames::getChannelNames( WLEModality::Enum type )
{
    if( m_namesMap->empty() )
    {
        return WLArrayList< std::string >::SPtr();
    }

    if( m_namesMap->count( type ) > 0 )
    {
        return m_namesMap->at( type );
    }

    return WLArrayList< std::string >::SPtr();
}

WLArrayList< std::string >::SPtr WFTChannelNames::getChannelNames()
{
    WLArrayList< std::string >::SPtr names( new WLArrayList< std::string > );

    for( ChanNamesMapT::iterator it = m_namesMap->begin(); it != m_namesMap->end(); ++it )
    {
        BOOST_FOREACH(std::string str, *it->second)
        {
            names->push_back( str );
        }
    }

    return names;
}

bool WFTChannelNames::fromFiff( MeasurementInfo_SPtr measInfo )
{
    if( getType() != WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES )
    {
        return false;
    }

    m_namesMap.reset( new ChanNamesMapT );

    if( measInfo == 0 )
    {
        return false;
    }

    for( int i = 0; i < measInfo->chs.size(); ++i )
    {

        WLEModality::Enum modality = WLEModality::fromFiffType( measInfo->chs.at( i ).kind );

        if( modality == WLEModality::UNKNOWN )
        {
            continue;
        }

        if( m_namesMap->count( modality ) == 0 )
        {
            m_namesMap->insert(
                            ChanNamesMapT::value_type( modality,
                                            WLArrayList< std::string >::SPtr( new WLArrayList< std::string > ) ) );
        }

        m_namesMap->at( modality )->push_back( measInfo->chs.at( i ).ch_name.toStdString() );
    }

    return true;
}
