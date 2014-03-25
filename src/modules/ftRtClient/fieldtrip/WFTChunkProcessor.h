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

#include <map>
#include <list>

#include <boost/shared_ptr.hpp>

#include "core/container/WLArrayList.h"
#include "core/data/emd/WLEMData.h"

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

    typedef boost::shared_ptr< WFTChunkProcessor > SPtr;

    /**
     * Destroys the WFTChunkProcessor.
     */
    virtual ~WFTChunkProcessor();

    /**
     * This method does a try to batch all chunks in the given collection. After processing you can receive the information using the getters.
     *
     * @param chunkList A pointer on a collection containing header chunks.
     * @return Return true if all chunks could be parsed, else false.
     */
    virtual bool process( WFTChunkList::SPtr chunkList );

    /**
     * Returns a WELMData object for the given FieldTrip channel number.
     *
     * @param channel The channel number in the FieldTrip buffer.
     * @return The WELMData object.
     */
    WLEMData::SPtr getWLEMDataType( size_t channel );

    /**
     * This method picks out the names of channels contained in the chunk list. Inside of a channel names chunk the values are represented as a '\0'
     * separated string. This implementation leaves the possibility open that the channel names comes in different chunks separately.
     *
     * @param chunkList A pointer on the list.
     * @return Returns a vector containing the found channel names.
     */
    virtual WLArrayList< std::string >::SPtr extractChannelNames( WFTChunkList::SPtr chunkList );

    /**
     * This method picks out the channel order and channel types included in the Channel Flags chunk.
     *
     * @param chunkList A list of header chunks.
     * @return Returns a list of strings.
     */
    virtual WLArrayList< std::string >::SPtr extractChannelFlags( WFTChunkList::SPtr chunkList );

private:

    /**
     * This method extracts the channel names from a single chunk.
     *
     * @param chunk A pointer on the chunk.
     * @param vector A reference on the resulting string vector.
     * @return Returns true if there was channel names in the chunk, else false.
     */
    bool getChannelNames( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr &vector );

    bool getChannelModality( WFTChunk::SPtr chunk, std::set< WLEModality::Enum >& modalities );

};

#endif /* WFTCHUNKPROCESSOR_H_ */
