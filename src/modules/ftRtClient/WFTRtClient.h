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
#include "io/dataTypes/WFTData.h"
#include "io/dataTypes/WFTHeader.h"
#include "io/dataTypes/WFTObject.h"
#include "io/response/WFTResponse.h"
#include "io/request/WFTRequest.h"

class WFTRtClient
{
public:

    static const std::string CLASS;

    static const UINT32_T DEFAULT_WAIT_TIMEOUT;

    typedef boost::shared_ptr< WFTRtClient > SPtr;

    WFTRtClient();

    virtual ~WFTRtClient();

    WFTConnection::SPtr getConnection() const;

    WFTHeader::SPtr getHeader() const;

    WFTData::SPtr getData() const;

    virtual void resetClient();

    void setConnection( WFTConnection::SPtr connection );

    bool connect();

    void disconnect();

    bool isConnected();

    bool hasNewSamples() const;

    bool hasNewEvents() const;

    UINT32_T getSampleCount() const;

    UINT32_T getEventCount() const;

    bool doRequest( WFTRequest& request, WFTResponse& response );

    bool doHeaderRequest();

    /**
     * Method to execute a Wait-Data_request on the buffer server. The first parameter will be filled with the current numbers of samples and events
     * from the server. The parameters two and three should contain the current numbers of your process. By default the clients values will be used,
     * so you has not to mention them. In case of only retrieving once samples or events you have to override the parameter you do not want to receive
     * with a very large integer value (e.g. 0xFFFFFFFF).
     *
     * @param samp_events The number of samples and events form the server.
     * @param samples Your current number of samples.
     * @param events Your current number of events.
     * @return Returns true whether the request was successful else false.
     */
    bool doWaitRequest( unsigned int samples = 0xFFFFFFFF, unsigned int events = 0xFFFFFFFF );

    /**
     * Method to receive new samples from the server. The method checks whether a request for new samples was executed before receiving. So you should do
     * a Wait-Request (recommend) or a Header-Request before calling this method.
     *
     * @return Returns true if the request was successful else false.
     */
    bool getNewSamples();

    bool getNewEvents();

    /**
     * This method transfers a FieldTrip response into a WFTObject derived data object defined by the type T.
     *
     * @param object The destination data object.
     * @param response The FieldTrip response.
     * @return Returns true if the transformation was successful else false.
     */
    template< typename T >
    bool getResponseAs( boost::shared_ptr< T > object, WFTResponse::SPtr response );

    UINT32_T getTimeout() const;

    void setTimeout(UINT32_T timeout);

protected:

    /**
     * The clients connection to the FieldTrip buffer server.
     */
    WFTConnection::SPtr m_connection;

    /**
     * A builder to create FieldTrip requests.
     */
    WFTRequestBuilder::SPtr m_reqBuilder;

    WFTHeader::SPtr m_ftHeader;

    WFTData::SPtr m_ftData;

    UINT32_T m_waitTimeout_ms;

private:

    /**
     * Variable to determine the number of received samples.
     */
    UINT32_T m_samples;

    /**
     * Variable to determine the number of received events.
     */
    UINT32_T m_events;

    WFTObject::WFTSamplesEventsT m_svr_samp_evt;
};

#endif /* WFTRTCLIENT_H_ */
