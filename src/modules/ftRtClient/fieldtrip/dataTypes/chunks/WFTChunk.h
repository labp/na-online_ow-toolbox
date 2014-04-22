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

#ifndef WFTCHUNK_H_
#define WFTCHUNK_H_

#include <ostream>

#include <boost/shared_ptr.hpp>

#include "modules/ftRtClient/fieldtrip/container/WLSmartStorage.h"
#include "modules/ftRtClient/fieldtrip/dataTypes/enum/WLEFTChunkType.h"
#include "modules/ftRtClient/fieldtrip/dataTypes/WFTObject.h"

/**
 * WFTChunk is a simple wrapper class for header chunks in a FieldTrip request. The Object administer the chunk
 * memory itself.
 * The class can be used in the WFTHeader chunk list. The header serializes the chunk object during the providing as
 * a put-request. In addition the WFTChunk or respective a chunk list can be iterated through a WFTChunkIterator.
 */
class WFTChunk: public WFTObject
{
public:

    /**
     * The chunk as a shared pointer.
     */
    typedef boost::shared_ptr< WFTChunk > SPtr;

    /**
     * A shared pointer on a constant chunk.
     */
    typedef boost::shared_ptr< const WFTChunk > ConstSPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WFTChunk.
     */
    WFTChunk();

    /**
     * The constructor defines the describing part of the chunk and allocates memory for the data.
     *
     * @param chunkType The type of the chunks data. This types values are specified as enum in <message.h> from FieldTrip.
     * @param chunkSize The number of bytes allocated by the data.
     * @param data A pointer to the data to saving. The constructor copies the data to its own memory area.
     */
    WFTChunk( UINT32_T chunkType, UINT32_T chunkSize, const void *data );

    /**
     * Constructs a new WFTChunk.
     *
     * @param chunkType The FieldTrip chunk type as unsigned integer.
     * @param data The data string.
     */
    WFTChunk( UINT32_T chunkType, const std::string data );

    /**
     * Constructs a new WFTChunk.
     *
     * @param chunkType The FieldTrip chunk type as WLEFTChunkType::Enum.
     * @param data The data string.
     */
    WFTChunk( WLEFTChunkType::Enum chunkType, const std::string data );

    /**
     * Returns the total size of the whole chunk including definition part and data part.
     * For getting the data size use "getDef().size" instead.
     *
     * This method is derived form WFTObject
     *
     * @return The total chunk object size.
     */
    UINT32_T getSize() const;

    /**
     * Gets the fix definition part of the chunk.
     *
     * @return The definition part.
     */
    const WFTChunkDefT getDef() const;

    /**
     * Returns the chunk type.
     *
     * @return The chunk type.
     */
    WLEFTChunkType::Enum getType() const;

    /**
     * Return a constant pointer to the chunks data content.
     */
    const void *getData() const;

    /**
     * The Method tries to create the chunks data as a std::string and returns the string.
     * If the chunk contains binary data of files they won't been printed as string.
     *
     * @return A std::string containing the data.
     */
    std::string getDataString() const;

protected:

    /**
     * The chunk object.
     */
    WFTChunkDefT m_chunkdef;

    /**
     * The variable chunk data area.
     */
    WLSmartStorage m_store;

};

inline std::ostream& operator<<( std::ostream& str, const WFTChunk& chunk )
{
    str << WFTChunk::CLASS << ":";
    str << " Type: " << chunk.getType();
    str << ", Size: " << chunk.getDef().size;
    str << ", Value: " << chunk.getDataString();

    return str;
}

#endif /* WFTCHUNK_H_ */
