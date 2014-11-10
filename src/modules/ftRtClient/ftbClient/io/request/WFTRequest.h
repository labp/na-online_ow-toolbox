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

#ifndef WFTREQUEST_H_
#define WFTREQUEST_H_

#include <ostream>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

#include "modules/ftRtClient/ftbClient/dataTypes/WFTObject.h"
#include "modules/ftRtClient/ftbClient/dataTypes/enum/WLEFTCommand.h"

/**
 * The WFTRequest class represents a basic FieldTrip request. It adapts the FieldTrip Buffer Request and can be used to
 * create new requests.
 */
class WFTRequest: public boost::enable_shared_from_this< WFTRequest >, protected FtBufferRequest
{
public:
    /**
     * A shared pointer on a WFTRequest.
     */
    typedef boost::shared_ptr< WFTRequest > SPtr;

    /**
     * A shared pointer on a constant WFTRequest.
     */
    typedef boost::shared_ptr< const WFTRequest > ConstSPtr;

    /**
     * Declare the << operator as friend.
     *
     * \param strm The input stream.
     * \param request The request object.
     *
     * \return Returns an output stream, which contains the request string.
     */
    friend std::ostream& operator<<( std::ostream &strm, const WFTRequest &request );

    /**
     * Creates a new WFTRequest.
     */
    WFTRequest();

    /**
     * Creates a new WFTRequest with an existing message.
     *
     * \param msg The message
     */
    explicit WFTRequest( const WFTMessageT *msg );

    /**
     * Destroys the WFTRequest.
     */
    virtual ~WFTRequest();

    /**
     * Gets the message header.
     *
     * \return The message header.
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
     * \return The messages content.
     */
    SimpleStorage &getBuffer();

    /**
     * Inherited method from FtBufferRequest.
     */
    FtBufferRequest::out;

    /**
     * Gets the abstract request as a concrete request.
     *
     * \return Returns a shared pointer on a concrete request.
     */
    template< typename Request >
    boost::shared_ptr< Request > getAs()
    {
        return boost::dynamic_pointer_cast< Request >( shared_from_this() );
    }

    /**
     * Gets the abstract request as a concrete request.
     *
     * \return Returns a shared pointer on a constant concrete request.
     */
    template< typename Request >
    boost::shared_ptr< const Request > getAs() const
    {
        return boost::dynamic_pointer_cast< Request >( shared_from_this() );
    }
};

inline std::ostream& operator<<( std::ostream &strm, const WFTRequest &request )
{
    strm << "WFTRequest: ";
    strm << "Version: " << request.m_def.version;
    strm << ", Command: " << WLEFTCommand::name( ( WLEFTCommand::Enum )request.m_def.command );
    strm << ", Buffer Size: " << request.m_def.bufsize;

    return strm;
}

#endif  // WFTREQUEST_H_
