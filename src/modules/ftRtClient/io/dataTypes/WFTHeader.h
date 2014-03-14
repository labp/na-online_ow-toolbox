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

#include <vector>

#include <boost/shared_ptr.hpp>

#include "WFTChunk.h"
#include "WFTRequestableObject.h"

class WFTHeader: public WFTObject
{

public:

    /**
     * A shared pointer on a WFTHeader.
     */
    typedef boost::shared_ptr< WFTHeader > SPtr;

    /**
     * A shared pointer on a list with WFTChunks.
     */
    typedef std::vector< WFTChunk::SPtr > WFTChunkList;

    /**
     * A shared pointer on a chunk list.
     */
    typedef boost::shared_ptr< WFTChunkList > WFTChunkList_SPtr;

    /**
     * A shared pointer on a immutable chunk list.
     */
    typedef boost::shared_ptr< const WFTChunkList > WFTChunkList_ConstSPtr;

    WFTHeader();

    WFTHeader( UINT32_T numChannels, UINT32_T dataType, float fsample );

    ~WFTHeader();

    WFTRequest::SPtr asRequest();

    bool parseResponse( WFTResponse::SPtr response );

    UINT32_T getSize() const;

    WFTHeaderDefT& getHeaderDef();

    bool hasChunks();

    void addChunk( WFTChunk::SPtr chunk );

    /**
     * Returns the chunks collection as shared pointer.
     *
     * @return A pointer on the chunk list.
     */
    WFTChunkList_ConstSPtr getChunks();

protected:

    /**
     * The definition part of the header.
     */
    WFTHeaderDefT m_def;

    /**
     * A list of chunk objects. It is used during request serializing.
     */
    WFTChunkList m_chunks;

};

#endif /* WFTHEADER_H_ */
