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

#ifndef WFTREQUEST_PUTHEADER_H_
#define WFTREQUEST_PUTHEADER_H_

#include <boost/shared_ptr.hpp>

#include "modules/ftRtClient/fieldtrip/dataTypes/chunks/WFTAChunk.h"
#include "WFTRequest.h"

class WFTRequest_PutHeader: public WFTRequest
{
public:

    /**
     * A pointer on a WFTRequest_PutHeader.
     */
    typedef boost::shared_ptr< WFTRequest_PutHeader > SPtr;

    WFTRequest_PutHeader( UINT32_T numChannels, UINT32_T dataType, float fsample );

    virtual ~WFTRequest_PutHeader();

    bool addChunk( UINT32_T chunkType, UINT32_T chunkSize, const void *data );

    bool addChunk( WFTAChunk::SPtr chunk );

};

#endif /* WFTREQUEST_PUTHEADER_H_ */
