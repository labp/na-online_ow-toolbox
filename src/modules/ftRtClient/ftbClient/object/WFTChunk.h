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

#ifndef WFTCHUNK_H_
#define WFTCHUNK_H_

#include <string>

#include "modules/ftRtClient/ftb/WFtbChunk.h"
#include "WFTObject.h"

/**
 * TODO(pieloth): documentation
 *
 * \author pieloth
 */
class WFTChunk : public WFTObject
{
public:
    /**
     * A shared pointer on a WFTChunk.
     */
    typedef boost::shared_ptr< WFTChunk > SPtr;

    /**
     * A shared pointer on a constant WFTChunk.
     */
    typedef boost::shared_ptr< const WFTChunk > ConstSPtr;

    static const std::string CLASS;

    WFTChunk( const wftb::ChunkDefT& chunkDef, const void* data );

    virtual ~WFTChunk();

    wftb::chunk_type_t getChunkType() const;

    const void* getData() const;

    virtual wftb::chunk_size_t getSize() const;

    virtual wftb::chunk_size_t getDataSize() const;

private:
    const wftb::chunk_type_t m_chunk_type;
    wftb::chunk_size_t m_dataSize;
    void* m_data;
};

#endif  // WFTCHUNK_H_
