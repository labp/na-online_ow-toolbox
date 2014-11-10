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

#include <iterator>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/range/adaptor/filtered.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm/copy.hpp>

#include "WFTChunkList.h"

bool WFTChunkList::isChunkType( const WFTAChunk::SPtr& chunk, wftb::chunk_type_t x )
{
    return chunk->getType() == x;
}

WFTChunkList::SPtr WFTChunkList::filter( wftb::chunk_type_t chunkType )
{
    WFTChunkList::SPtr q( new WFTChunkList() );
    WFTChunkList &ptr = *q;

    // use boost range iterator to filter for the given chunk type
    using boost::adaptors::filtered;
    WFTChunkList &x = *this;
    boost::copy( x | filtered( bind( &WFTChunkList::isChunkType, _1, chunkType ) ), std::back_inserter( ptr ) );

    return q;
}
