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
#include <string>

#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

#include "modules/ftRtClient/ftb/WFtBuffer.h"
#include "modules/ftRtClient/ftb/WFtbCommand.h"

/**
 * The WFTRequest class represents a basic FieldTrip request. It adapts the FieldTrip Buffer Request and can be used to
 * create new requests.
 */
class WFTRequest: public FtBufferRequest
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

    static const std::string CLASS;

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
    explicit WFTRequest( const wftb::MessageT *msg );

    /**
     * Destroys the WFTRequest.
     */
    virtual ~WFTRequest();

    /**
     * Gets the message header.
     *
     * \return The message header.
     */
    wftb::MessageDefT& getMessageDef();

    /**
     * Gets the message.
     *
     * \return The message.
     */
    wftb::MessageT& getMessage();

    /**
     * Inherited method from FtBufferRequest.
     */
    FtBufferRequest::out;
};

inline std::ostream& operator<<( std::ostream &strm, const WFTRequest &request )
{
    strm << WFTRequest::CLASS << ": ";
    strm << "version=" << request.m_def.version;
    strm << ", command=" << wftb::CommandType::name( request.m_def.command );
    strm << ", bufsize=" << request.m_def.bufsize;

    return strm;
}

#endif  // WFTREQUEST_H_
