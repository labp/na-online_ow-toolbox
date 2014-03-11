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

#ifndef WFTRTCLIENT_H_
#define WFTRTCLIENT_H_

#include <boost/shared_ptr.hpp>

#include <message.h>

#include "connection/WFTConnection.h"
#include "io/WFTRequestBuilder.h"
#include "io/dataTypes/WFTObject.h"
#include "io/response/WFTResponse.h"

class WFTRtClient
{
public:

    static const std::string CLASS;

    typedef boost::shared_ptr< WFTRtClient > SPtr;

    WFTRtClient( WFTConnection::SPtr connection );

    WFTConnection::SPtr getConnection() const;

    bool connect();

    void disconnect();

    bool isConnected();

    void doReqest();

    /**
     * This method transfers a FieldTrip response into a WFTObject derived data object defined by the type T.
     *
     * @param object The destination data object.
     * @param response The FieldTrip response.
     * @return Returns true if the transformation was successful else false.
     */
    template< typename T >
    bool getResponseAs( boost::shared_ptr<T> object, WFTResponse::SPtr response );

protected:

    /**
     * The clients connection to the FieldTrip buffer server.
     */
    WFTConnection::SPtr m_connection;

    /**
     * A builder to create FieldTrip requests.
     */
    WFTRequestBuilder::SPtr m_reqBuilder;

};

#endif /* WFTRTCLIENT_H_ */
