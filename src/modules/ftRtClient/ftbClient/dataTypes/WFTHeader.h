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

#ifndef WFTHEADER_H_
#define WFTHEADER_H_

#include <ostream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/container/WLList.h"

#include "modules/ftRtClient/ftb/WFtBuffer.h"
#include "modules/ftRtClient/ftb/WFtbChunk.h"
#include "modules/ftRtClient/ftb/WFtbData.h"
#include "modules/ftRtClient/ftbClient/dataTypes/WFTRequestableObject.h"
#include "modules/ftRtClient/ftbClient/response/WFTResponse.h"
#include "modules/ftRtClient/ftbClient/request/WFTRequest.h"
#include "WFTChunk.h"

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
     * \param numChannels The number of channels.
     * \param dataType The used data type in the data buffer. The parameters data type is defined by the FieldTrip buffer protocol.
     * \param fsample The sample frequency.
     */
    WFTHeader( wftb::nchans_t numChannels, wftb::data_type_t dataType, wftb::fsamp_t fsample );

    /**
     * Destroys the WFTHeader.
     */
    virtual ~WFTHeader();

    /**
     * Method to initialize the WFTHeader during constructor call.
     *
     * \param numChannels The number of channels.
     * \param dataType The used data type in the data buffer. The parameters data type is defined by the FieldTrip buffer protocol.
     * \param fsample The sample frequency.
     */
    void init( wftb::nchans_t numChannels, wftb::data_type_t dataType, wftb::fsamp_t fsample );

    /**
     * Inherited form WFTRequestableObject.
     *
     * \return The header as request object.
     */
    WFTRequest::SPtr asRequest();

    /**
     * Inherited form WFTRequestableObject.
     *
     * \param response The FieldTrip response to parse.
     * \return Returns true if the parsing was successful, else false.
     */
    bool parseResponse( WFTResponse::SPtr response );

    /**
     * Inherited form WFTObject.
     *
     * \return Returns the amount of memory allocated by the header including the size of the chunk list.
     */
    wftb::bufsize_t getSize() const;

    /**
     * Returns a reference on the headers definition part.
     *
     * \return The header definition.
     */
    wftb::HeaderDefT& getHeaderDef();

    /**
     * Returns the header structure.
     *
     * \return The header structure.
     */
    wftb::HeaderDefT getHeaderDef() const;

    /**
     * Returns whether or not the header has chunks in its buffer.
     *
     * \return Returns true if there are chunks, else false.
     */
    bool hasChunks() const;

    /**
     * Returns whether or not the header has chunks of a specific chunk type in its buffer.
     *
     * \param chunkType The chunk type to filter.
     * \return Returns true if there are chunks kind of the @chunkType, else false.
     */
    bool hasChunk( wftb::chunk_type_t chunkType ) const;

    /**
     * Add a new chunk to the headers chunk list.
     *
     * \param chunk The new chunk.
     */
    void addChunk( WFTChunk::SPtr chunk );

    /**
     * Returns the chunks collection as shared pointer.
     *
     * \return A pointer on the chunk list.
     */
    WLList< WFTChunk::SPtr >::ConstSPtr getChunks() const;

protected:
    /**
     * The definition part of the header.
     */
    wftb::HeaderDefT m_def;

    /**
     * A list of chunk objects. It is used during request serializing.
     */
    WLList< WFTChunk::SPtr >::SPtr m_chunks;
};

inline std::ostream& operator<<( std::ostream& str, const WFTHeader& header )
{
    str << WFTHeader::CLASS << ":";
    str << " DataType: " << header.getHeaderDef().data_type;
    str << ", Sample Frequency:" << header.getHeaderDef().fsample;
    str << ", Channels: " << header.getHeaderDef().nchans;
    str << ", Samples: " << header.getHeaderDef().nsamples;
    str << ", Events: " << header.getHeaderDef().nevents;
    str << ", Buffer Size: " << header.getHeaderDef().bufsize;
    header.hasChunks() ? str << ", Chunks: " << header.getChunks()->size() : str << ", Chunks: 0";

    return str;
}

#endif  // WFTHEADER_H_
