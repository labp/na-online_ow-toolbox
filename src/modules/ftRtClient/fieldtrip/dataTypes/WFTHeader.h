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

#ifndef WFTHEADER_H_
#define WFTHEADER_H_

#include <ostream>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <fiff/fiff_info.h>

#include "core/container/WLArrayList.h"

#include "WFTChunk.h"
#include "WFTChunkList.h"
#include "WFTRequestableObject.h"

/**
 * The WFTHeader class represents the FieldTrip header structure. It consists of a fix definition part and the chunk list collection.
 */
class WFTHeader: public WFTRequestableObject
{

public:

    /**
     * A shared pointer on a WFTHeader.
     */
    typedef boost::shared_ptr< WFTHeader > SPtr;

    /**
     * A pointer on the measurement information.
     */
    typedef boost::shared_ptr< FIFFLIB::FiffInfo > MeasurementInfo_SPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructor to create a new WFTHeader.
     */
    WFTHeader();

    /**
     * Constructor to create a new header structure.
     *
     * @param numChannels The number of channels.
     * @param dataType The used data type in the data buffer. The parameters data type is defined by the FieldTrip buffer protocol.
     * @param fsample The sample frequency.
     */
    WFTHeader( UINT32_T numChannels, UINT32_T dataType, float fsample );

    /**
     * Method to initialize the WFTHeader during constructor call.
     *
     * @param numChannels The number of channels.
     * @param dataType The used data type in the data buffer. The parameters data type is defined by the FieldTrip buffer protocol.
     * @param fsample The sample frequency.
     */
    void init( UINT32_T numChannels, UINT32_T dataType, float fsample );

    /**
     * Inherited form WFTRequestableObject.
     *
     * @return The header as request object.
     */
    WFTRequest::SPtr asRequest();

    /**
     * Inherited form WFTRequestableObject.
     *
     * @param response The FieldTrip response to parse.
     * @return Returns true if the parsing was successful, else false.
     */
    bool parseResponse( WFTResponse::SPtr response );

    /**
     * Inherited form WFTObject.
     *
     * @return Returns the amount of memory allocated by the header including the size of the chunk list.
     */
    UINT32_T getSize() const;

    /**
     * Returns a reference on the headers definition part.
     *
     * @return The header definition.
     */
    WFTHeaderDefT& getHeaderDef();

    /**
     * Returns the header structure.
     *
     * @return The header structure.
     */
    WFTHeaderDefT getHeaderDef() const;

    /**
     * Returns whether or not the header has chunks in its buffer.
     *
     * @return Returns true if there are chunks, else false.
     */
    bool hasChunks() const;

    /**
     * Returns whether or not the header has chunks of a specific chunk type in its buffer.
     *
     * @param chunkType The chunk type to filter.
     * @return Returns true if there are chunks kind of the @chunkType, else false.
     */
    bool hasChunk( WLEFTChunkType::Enum chunkType ) const;

    /**
     * Add a new chunk to the headers chunk list.
     *
     * @param chunk The new chunk.
     */
    void addChunk( WFTChunk::SPtr chunk );

    /**
     * Returns the chunks collection as shared pointer.
     *
     * @return A pointer on the chunk list.
     */
    WFTChunkList::ConstSPtr getChunks() const;

    /**
     * Returns the chunks collection filtered for a specific @chunkType as shared pointer.
     * The returned pointers target is just a copy of the original collection.
     *
     * @param chunkType The chunk type.
     * @return A pointer on the chunk list.
     */
    WFTChunkList::SPtr getChunks( WLEFTChunkType::Enum chunkType );

    /**
     * Gets the measurement information.
     *
     * @return The measurement information.
     */
    MeasurementInfo_SPtr getMeasurementInfo();

    /**
     * Sets the measurement information.
     *
     * @param info The measurement information.
     */
    void setMeasurementInfo( MeasurementInfo_SPtr info );

protected:

    /**
     * The definition part of the header.
     */
    WFTHeaderDefT m_def;

    /**
     * A list of chunk objects. It is used during request serializing.
     */
    boost::shared_ptr< WFTChunkList > m_chunks;

    /**
     * The measurement information.
     */
    boost::shared_ptr< FIFFLIB::FiffInfo > m_measurementInfo;

};

inline std::ostream& operator<<( std::ostream& str, const WFTHeader& header )
{
    str << WFTHeader::CLASS << ":";
    str << " DataType: " << header.getHeaderDef().data_type;
    str << ", Sample Frq:" << header.getHeaderDef().fsample;
    str << ", Channels: " << header.getHeaderDef().nchans;
    str << ", Samples: " << header.getHeaderDef().nsamples;
    str << ", Events: " << header.getHeaderDef().nevents;
    str << ", Buffer Size: " << header.getHeaderDef().bufsize;
    header.hasChunks() ? str << ", Chunks: " << header.getChunks()->size() : str << ", Chunks: 0";

    return str;
}

#endif /* WFTHEADER_H_ */
