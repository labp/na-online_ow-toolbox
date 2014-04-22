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

#ifndef WFTCHUNKITERATOR_H_
#define WFTCHUNKITERATOR_H_

#include <boost/shared_ptr.hpp>

#include "dataTypes/chunks/WFTChunk.h"
#include "WFTAIterator.h"

/**
 * The WFTChunkIterator can be used to run through a bulk of memory containing FieldTrip chunks.
 * This class has the standard iterator appearance with its characteristic operations.
 */
class WFTChunkIterator: public WFTAIterator< WFTChunk >
{
public:

    /**
     * A shared pointer on the iterator.
     */
    typedef boost::shared_ptr< WFTChunkIterator > SPtr;

    /**
     * The constructor defines the chunk storage for followed iterations.
     *
     * @param buf A reference to the chunk storage memory.
     * @param size The memory size allocated by all chunks together.
     */
    WFTChunkIterator( SimpleStorage& buf, int size );

    /**
     * Inherited method from WFTAIterator.
     *
     * @return Returns true if there are more chunks, else false.
     */
    bool hasNext() const;

    /**
     * Inherited method from WFTAIterator.
     *
     * @return Returns the next chunk element.
     */
    WFTChunk::SPtr getNext();

};

#endif /* WFTCHUNKITERATOR_H_ */
