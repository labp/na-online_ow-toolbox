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

#ifndef WFTCHUNKPROCESSOR_H_
#define WFTCHUNKPROCESSOR_H_

#include "core/container/WLArrayList.h"

#include "dataTypes/WFTChunk.h"
#include "dataTypes/WFTHeader.h"
#include "dataTypes/WFTChunkList.h"

/**
 * The WFTChunkProcessor class is created to apply some chunk specific operations on the FieldTrip header chunks managed in the WFTHeader class.
 * This is the right place to extend the implementation by chunk operations through inherit the WFTChunkProcessor.
 */
class WFTChunkProcessor
{
public:

    /**
     * Destroys the WFTChunkProcessor.
     */
    virtual ~WFTChunkProcessor();

    /**
     * This method picks out the names of channels contained in the chunk list. Inside of a channel names chunk the values are represented as a '\0'
     * separated string. This implementation leaves the possibility open that the channel names comes in different chunks separately.
     *
     * @param chunkList A pointer on the list.
     * @return Returns a vector containing the found channel names.
     */
    virtual WLArrayList< std::string >::SPtr extractChannelNames( WFTChunkList::SPtr chunkList );

private:

    /**
     * This method extracts the channel names from a single chunk.
     *
     * @param chunk A pointer on the chunk.
     * @param vector A reference on the resulting string vector.
     * @return Returns true if there was channel names in the chunk, else false.
     */
    virtual bool getChannelNames( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr &vector );
};

#endif /* WFTCHUNKPROCESSOR_H_ */
