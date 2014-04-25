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

#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>

#include <core/common/WLogger.h>

#include "WFTChunkChanNames.h"

WFTChunkChanNames::WFTChunkChanNames( const char* data, const size_t size ) :
                WFTAChunk( data, size )
{
    processData( data, size );
}

WLEFTChunkType::Enum WFTChunkChanNames::getType() const
{
    return WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES;
}

WLArrayList< std::string >::ConstSPtr WFTChunkChanNames::getData( const WLEModality::Enum modality ) const
{
    if( m_namesMap->count( modality ) > 0 )
    {
        return m_namesMap->at( modality );
    }

    return WLArrayList< std::string >::ConstSPtr();
}

bool WFTChunkChanNames::process( const char* data, size_t size )
{
    m_namesMap.reset( new ChanNamesMapT );
    std::vector< std::string > splitVec;
    std::string str( data, size );

    boost::split( splitVec, str, boost::is_any_of( "\0" ), boost::token_compress_on );

    if( splitVec.size() == 0 )
    {
        return false;
    }

    BOOST_FOREACH(std::string chanName, splitVec)
    {
        WLEModality::Enum modality = WLEModality::UNKNOWN;
        std::string channel = chanName;
        boost::algorithm::to_lower( channel );
        std::pair< std::string, WLEModality::Enum > label;

        BOOST_FOREACH(label, m_modalityLabels)
        {
            std::size_t found = channel.find( label.first );
            if( found != std::string::npos )
            {
                modality = label.second;
                break;
            }
        }

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

        m_namesMap->at( modality )->push_back( chanName );
    }

    return m_namesMap->size() > 0;
}

void WFTChunkChanNames::insertLabels()
{
    m_modalityLabels.insert( ChanNameLabelT::value_type( "eeg", WLEModality::EEG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "meg", WLEModality::MEG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "eog", WLEModality::EOG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "ecg", WLEModality::ECG ) );
}
