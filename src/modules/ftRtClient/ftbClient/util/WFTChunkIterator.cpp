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

#include "modules/ftRtClient/ftb/WFtbChunk.h"

#include "WFTChunkIterator.h"

WFTChunkIterator::WFTChunkIterator( SimpleStorage* const buf, int size ) :
                WFTAIterator< WFTChunk >( buf, size )
{
}

bool WFTChunkIterator::hasNext() const
{
    return m_pos + ( int )sizeof(wftb::ChunkDefT) < m_size;
}

WFTChunk::SPtr WFTChunkIterator::getNext()
{
    if( !hasNext() ) // when end arrived at the end, return an empty pointer.
    {
        return WFTChunk::SPtr();
    }

    wftb::ChunkDefT* chunkdef = ( wftb::ChunkDefT* )( ( char * )m_store->data() + m_pos );
    // when the chunk has no data, set position to next chunk and return. (should just happens in case of an error)
    if( chunkdef->size == 0 )
    {
        m_pos += sizeof(wftb::ChunkDefT);
        return WFTChunk::SPtr();
    }

    char *src = ( ( char * )m_store->data() ) + m_pos + sizeof(wftb::ChunkDefT); // pointer to the chunks data.
    m_pos += sizeof(wftb::ChunkDefT) + chunkdef->size; // start position for the next chunk (next definition).

    return WFTChunk::SPtr( new WFTChunk( *chunkdef, src ) );
}
