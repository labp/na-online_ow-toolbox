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

using namespace std;

const std::string WFTChunkChanNames::CLASS = "WFTChunkChanNames";

WFTChunkChanNames::WFTChunkChanNames( const char* data, const size_t size ) :
                WFTAChunk( WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES, size )
{
    insertLabels();

    processData( data, size );
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
    wlog::debug( CLASS ) << "process() called.";

    m_namesMap.reset( new ChanNamesMapT );
    std::vector< std::string > splitVec;
    std::string str( data, size );
    int chans = 0;

    split( str, splitVec, '\0' );

    if( splitVec.size() == 0 )
    {
        return false;
    }

    BOOST_FOREACH(std::string chanName, splitVec)
    {
        WLEModality::Enum modality = WLEModality::UNKNOWN;
        bool found = false;
        std::string channel( chanName );
        boost::algorithm::to_lower( channel );
        std::pair< std::string, WLEModality::Enum > label;

        BOOST_FOREACH(label, m_modalityLabels)
        {
            found = channel.find( label.first ) != std::string::npos;
            if( found )
            {
                modality = label.second;
                break;
            }
        }

        if( !found )
        {
            wlog::debug( CLASS ) << "Reject channel name: " << chanName;
            continue;
        }

        if( m_namesMap->count( modality ) == 0 )
        {
            m_namesMap->insert(
                            ChanNamesMapT::value_type( modality,
                                            WLArrayList< std::string >::SPtr( new WLArrayList< std::string >() ) ) );
        }

        m_namesMap->at( modality )->push_back( chanName );
        ++chans;
    }

    wlog::debug( CLASS ) << "Channel names read. Number of assigned channels: " << chans;
    wlog::debug( CLASS ) << "Channel names in string vector: " << splitVec.size();
    std::pair< WLEModality::Enum, WLArrayList< std::string >::SPtr > list;
    BOOST_FOREACH(list, *m_namesMap)
    {
        wlog::debug( CLASS ) << "Channel names for modality " << list.first << ": " << list.second->size();
    }

    return m_namesMap->size() > 0;
}

void WFTChunkChanNames::insertLabels()
{
    m_modalityLabels.insert( ChanNameLabelT::value_type( "eeg", WLEModality::EEG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "meg", WLEModality::MEG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "eog", WLEModality::EOG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "ecg", WLEModality::ECG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "misc", WLEModality::UNKNOWN ) );
}

WLSmartStorage::ConstSPtr WFTChunkChanNames::serialize() const
{
    WLSmartStorage::SPtr store( new WLSmartStorage );

    for( ChanNamesMapT::iterator it = m_namesMap->begin(); it != m_namesMap->end(); ++it )
    {
        BOOST_FOREACH(std::string channel, *it->second)
        {
            store->append( channel );
        }
    }

    return store;
}

std::vector< std::string >& WFTChunkChanNames::split( std::string str, std::vector< std::string >& result, const char delim )
{
    result.clear();

    size_t end = -1;
    size_t start = 0;

    int counter = 0;

    BOOST_FOREACH(char ch, str)
    {
        ++end;

        std::string cmp_str( &ch, 1 );

        if( ch == delim )
        {
            ++counter;

            result.push_back( str.substr( start, end - start ) );
            start = end + 1;
        }
    }

    return result;
}
