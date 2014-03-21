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

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>

#include "dataTypes/WLEFTChunkType.h"

#include "WFTChunkProcessor.h"

WLArrayList< std::string >::SPtr WFTChunkProcessor::extractChannelNames( WFTChunkList::SPtr chunkList )
{
    WLArrayList< std::string >::SPtr vector( new WLArrayList< std::string >() );

    BOOST_FOREACH(WFTChunk::SPtr chunk, *(chunkList->filter(WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES)))
    {
        getChannelNames( chunk, vector );
    }

    return WLArrayList< std::string >::SPtr( vector );
}

bool getChannelNames( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr &vector )
{
    if( chunk->getType() != WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES )
        return false;

    std::vector< std::string > splitVec;
    std::string str = chunk->getDataString();

    boost::split( splitVec, str, boost::is_any_of( "\0" ), boost::token_compress_on );

    /**
     * TODO(LaBP): After splitting the channel name string into its elements based on the "\0" delimiter, extract the single channel names after reviewing he result from Neuromag.
     */
    /*
     if(splitVec.size() < 2)
     return StringVector::SPtr();
     */

    if( splitVec.size() == 0 )
    {
        return false;
    }

    BOOST_FOREACH(std::string chanName, splitVec)
    {
        vector->push_back( chanName );
    }

    return vector->size() > 0;
}
