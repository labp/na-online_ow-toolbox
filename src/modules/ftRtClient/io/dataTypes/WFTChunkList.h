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

#ifndef WFTCHUNKLIST_H_
#define WFTCHUNKLIST_H_

#include <boost/shared_ptr.hpp>

#include <core/container/WLArrayList.h>

#include "WLEFTChunkType.h"
#include "WFTChunk.h"

/**
 * The WFTChunkList class provides a vector associated container structure for WFTChunk types.
 */
class WFTChunkList: public WLArrayList< WFTChunk::SPtr >
{

public:

    /**
     * Shared pointer on a chunk list.
     */
    typedef boost::shared_ptr< WFTChunkList > SPtr;

    /**
     * Constant shared  pointer on a chunk list.
     */
    typedef boost::shared_ptr< const WFTChunkList > ConstSPtr;

    /**
     * This static method detects whether the @chunk is from the wished chunk type @type.
     *
     * @param chunk The chunk object.
     * @param type The chunk type
     * @return Returns true if the @chunk has the given @type, else false.
     */
    static bool isChunkType( const WFTChunk::SPtr& chunk, WLEFTChunkType::Enum type );

    /**
     * This method delivers a filtered chunk list in which all chunks has the given @chunkType.
     * The returning list is just a copy of the original collection.
     *
     * @param chunkType The type to filter
     * @return Returns a shared pointer on the filtered list.
     */
    WFTChunkList::SPtr filter( WLEFTChunkType::Enum chunkType );

};

#endif /* WFTCHUNKLIST_H_ */
