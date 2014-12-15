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

#include <boost/shared_ptr.hpp>

#include "core/container/WLList.h"

#include "modules/ftRtClient/ftb/WFtBuffer.h"
#include "modules/ftRtClient/ftb/WFtbChunk.h"
#include "modules/ftRtClient/ftb/WFtbData.h"

#include "../network/WFTResponse.h"
#include "WFTChunk.h"
#include "WFTDeserializable.h"
#include "WFTObject.h"

/**
 * The WFTHeader class represents the FieldTrip header structure. It consists of a fix definition part and the chunk list collection.
 */
class WFTHeader: public WFTDeserializable< WFTResponse >, public WFTObject
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
     * Destroys the WFTHeader.
     */
    virtual ~WFTHeader();

    virtual bool deserialize( const WFTResponse& response );

    virtual wftb::bufsize_t getSize() const;

    virtual wftb::bufsize_t getDataSize() const;

    /**
     * Returns a const reference on the headers definition part.
     *
     * \return The header structure.
     */
    const wftb::HeaderDefT& getHeaderDef() const;

    /**
     * Returns the chunks collection as shared pointer.
     *
     * \return A pointer on the chunk list.
     */
    WLList< WFTChunk::SPtr >::ConstSPtr getChunks() const;

private:
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
    str << WFTHeader::CLASS << ": ";
    str << " data_type=" << wftb::DataType::name( header.getHeaderDef().data_type );
    str << ", fsample=" << header.getHeaderDef().fsample;
    str << ", nchans=" << header.getHeaderDef().nchans;
    str << ", nsamples=" << header.getHeaderDef().nsamples;
    str << ", nevents=" << header.getHeaderDef().nevents;
    str << ", bufsize=" << header.getHeaderDef().bufsize;
    str << ", chunks=" << header.getChunks()->size();
    return str;
}

#endif  // WFTHEADER_H_
