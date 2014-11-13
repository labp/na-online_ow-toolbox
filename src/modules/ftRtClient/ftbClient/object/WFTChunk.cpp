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

#include <cstdlib>  // malloc
#include <cstring>  // memcpy

#include <core/common/WLogger.h>

#include "WFTChunk.h"

const std::string WFTChunk::CLASS = "WFTChunk";

WFTChunk::WFTChunk( const wftb::ChunkDefT& chunkDef, const void* data ) :
                m_chunk_type( chunkDef.type )
{
    m_dataSize = chunkDef.size;
    if( m_dataSize > 0 && data != NULL )
    {
        // TODO(pieloth): Do we need this copy?
        m_data = malloc( m_dataSize );
        memcpy( m_data, data, m_dataSize );
    }
    else
    {
        wlog::warn( CLASS ) << "Chunk contains no data!";
        m_dataSize = 0;
        m_data = NULL;
    }
}

WFTChunk::~WFTChunk()
{
    if( m_data != NULL )
    {
        free( m_data );
    }
}

wftb::chunk_type_t WFTChunk::getChunkType() const
{
    return m_chunk_type;
}

const void* WFTChunk::getData() const
{
    return m_data;
}

wftb::chunk_size_t WFTChunk::getSize() const
{
    return m_dataSize + sizeof(wftb::ChunkDefT);
}

wftb::chunk_size_t WFTChunk::getDataSize() const
{
    return m_dataSize;
}
