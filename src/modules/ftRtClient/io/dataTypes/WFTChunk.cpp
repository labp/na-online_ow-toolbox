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

#include "WFTChunk.h"

WFTChunk::WFTChunk( UINT32_T chunkType, UINT32_T chunkSize, const void *data )
{
    if( !m_buf.resize( chunkSize ) )
    {
        return;
    }

    memcpy( m_buf.data(), data, chunkSize );

    m_chunkdef.type = chunkType;
    m_chunkdef.size = chunkSize;
}

WFTChunk::~WFTChunk()
{
}

UINT32_T WFTChunk::getSize() const
{
    return m_chunkdef.size + sizeof(WFTChunkDefT);
}

const WFTObject::WFTChunkDefT WFTChunk::getDef() const
{
    return m_chunkdef;
}

WLEFTChunkType::Enum WFTChunk::getType() const
{
    return ( WLEFTChunkType::Enum )m_chunkdef.type;
}

const void *WFTChunk::getData()
{
    return m_buf.data();
}

const std::string WFTChunk::getDataString()
{
    char *chr = ( char* )malloc( m_chunkdef.size );
    memcpy( chr, m_buf.data(), m_chunkdef.size );

    std::string s = std::string( chr );

    delete chr;

    return s;
}
