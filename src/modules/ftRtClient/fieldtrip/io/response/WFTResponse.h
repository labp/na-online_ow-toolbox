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

#ifndef WFTRESPONSE_H_
#define WFTRESPONSE_H_

#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

/**
 * Wrapper class for a response created through a FieldTrip request.
 */
class WFTResponse: public FtBufferResponse
{
public:

    /**
     * Define a shared pointer on a response.
     */
    typedef boost::shared_ptr< WFTResponse > SPtr;

    typedef boost::shared_ptr< const message_t > WFTMessageT_ConstSPtr;

    /**
     * This method tests the arrived data for mistakes. It should be called before getting any data for the response.
     *
     * @return True if the data are valid, else false.
     */
    bool isValid() const;

    bool hasData() const;

    WFTMessageT_ConstSPtr getMessage();

protected:

    /**
     * Forbid direct access to the buffer space. Use getMessage() instead.
     */
    FtBufferResponse::m_response;
};

#endif /* WFTRESPONSE_H_ */
