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

#include "WFTAChunk.h"

WFTAChunk::WFTAChunk( WLEFTChunkType::Enum type, const size_t size ) :
                m_valid( false ), m_size( size ), m_type( type )
{
}

WFTAChunk::~WFTAChunk()
{
}

bool WFTAChunk::isValid() const
{
    return m_valid;
}

size_t WFTAChunk::getSize() const
{
    return m_size;
}

WLEFTChunkType::Enum WFTAChunk::getType() const
{
    return m_type;
}

void WFTAChunk::processData( const char* data, const size_t size )
{
    m_valid = process( data, size );
}
