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

#ifndef WFTREQUEST_H_
#define WFTREQUEST_H_

#include <ostream>

#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

#include "modules/ftRtClient/fieldtrip/dataTypes/WFTObject.h"
#include "modules/ftRtClient/fieldtrip/dataTypes/enum/WLEFTCommand.h"

/**
 * The WFTRequest class represents a basic FieldTrip request. It adapts the FieldTrip Buffer Request and can be used to
 * create new requests.
 */
class WFTRequest: protected FtBufferRequest
{
public:

    /**
     * A shared pointer on a WFTRequest.
     */
    typedef boost::shared_ptr< WFTRequest > SPtr;

    friend std::ostream& operator<<( std::ostream &strm, const WFTRequest &request );

    /**
     * Creates a new WFTRequest.
     */
    WFTRequest();

    /**
     * Creates a new WFTRequest with an existing message.
     *
     * @param msg The message
     */
    WFTRequest( const WFTMessageT *msg );

    /**
     * Gets the message header.
     *
     * @return The message header.
     */
    WFTMessageDefT &getMessageDef();

    /**
     * Gets the message.
     *
     * @return The message.
     */
    WFTMessageT &getMessage();

    /**
     * Gets the messages content.
     *
     * @return The messages content.
     */
    SimpleStorage &getBuffer();

    /**
     * Inherited method from FtBufferRequest.
     */
    FtBufferRequest::out;

};

inline std::ostream& operator<<( std::ostream &strm, const WFTRequest &request )
{
    strm << "WFTRequest: ";
    strm << "Version: " << request.m_def.version;
    strm << ", Command: " << WLEFTCommand::name( ( WLEFTCommand::Enum )request.m_def.command );
    strm << ", Buffer Size: " << request.m_def.bufsize;

    return strm;
}

#endif /* WFTREQUEST_H_ */
