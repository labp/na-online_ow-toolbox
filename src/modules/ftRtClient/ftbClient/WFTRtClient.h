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

#ifndef WFTRTCLIENT_H_
#define WFTRTCLIENT_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <message.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMDRaw.h"

#include "modules/ftRtClient/ftb/WFtBuffer.h"
#include "modules/ftRtClient/ftb/WFtbCommand.h"
#include "dataTypes/WFTData.h"
#include "dataTypes/WFTEventList.h"
#include "dataTypes/WFTHeader.h"
#include "dataTypes/WFTObject.h"
#include "io/request/WFTRequest.h"
#include "io/response/WFTResponse.h"
#include "WFTConnection.h"
#include "WFTRequestBuilder.h"

/**
 * WFTRtClient is the basic client interface to FieldTrip. It provides the essential buffer operations (e.g. requests and connections).
 * For simple I/O processing with the FieldTrip buffer server this class can be used. In case of additional processing on the retrieving
 * raw data you should prefer to inherit this client structure.
 */
class WFTRtClient
{
public:
    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Default timeout value for Wait-requests.
     */
    static const wftb::time_t DEFAULT_WAIT_TIMEOUT;

    /**
     * Shared pointer on WFTRtClient.
     */
    typedef boost::shared_ptr< WFTRtClient > SPtr;

    /**
     * Constructs the WFTRtClient by initializing the members and setting default values.
     */
    WFTRtClient();

    /**
     * Destroys the WFTRtClient.
     */
    virtual ~WFTRtClient();

    /**
     * Gets the clients WFTConnection object.
     *
     * \return The WFTConnection object.
     */
    WFTConnection::SPtr getConnection() const;

    /**
     * Gets FieldTrips header structure.
     *
     * \return The WFTHeader object.
     */
    WFTHeader::SPtr getHeader() const;

    /**
     * Gets FieldTrips data structure.
     *
     * \return The WFTData object.
     */
    WFTData::SPtr getData() const;

    /**
     * Gets the FieldTrip events structure.
     *
     * \return The FieldTrip events structure.
     */
    WFTEventList::SPtr getEventList() const;

    /**
     * Resets the client by setting the local members to their default values.
     * This method can be inherited for additional reset operations.
     */
    virtual void resetClient();

    /**
     * Sets the clients WFTConnection object.
     *
     * \param connection The WFTConnection object.
     */
    void setConnection( WFTConnection::SPtr connection );

    /**
     * Does a try to connect the client to the buffer server with the configured connection.
     *
     * \return Returns true if the connection could be established, else false.
     */
    bool connect();

    /**
     * Disconnects the client from the server.
     */
    void disconnect();

    /**
     * Gets whether the client has an open connection to the buffer server.
     *
     * \return Returns true if the there is an open connection, else false.
     */
    bool isConnected();

    /**
     * Gets whether new sample data arrived the client.
     *
     * \return Returns true if there are new samples, else false.
     */
    bool hasNewSamples() const;

    /**
     * Gets whether new events arrived the client.
     *
     * \return Returns true if there are new events, else false.
     */
    bool hasNewEvents() const;

    /**
     * Gets the number of read samples by the client.
     *
     * \return The number of read samples.
     */
    wftb::nsamples_t getSampleCount() const;

    /**
     * Gets the number of read events by the client.
     *
     * \return The number of read events.
     */
    wftb::nevents_t getEventCount() const;

    /**
     * Using this method you can put a single request on the server using a TCP request. Every further complex request comes down
     * to this method.
     * After requesting the servers answer structure will be filled into the @response parameter.
     *
     * \param request The request object.
     * \param response The response object.
     *
     * \return Returns true if the request was successful and the response could be read, else false.
     */
    bool virtual doRequest( WFTRequest::ConstSPtr request, WFTResponse::SPtr response );

    /**
     * This method does a header request and stores the resulting data in the local member, which can be accessed by getHeader().
     *
     * \return Returns true if the request was successful, else false.
     */
    bool virtual doHeaderRequest();

    /**
     * Method to execute a Wait-Data_request on the buffer server. The first parameter will be filled with the current numbers of samples and events
     * from the server. The parameters two and three should contain the current numbers of your process. By default the parameter for samples and
     * events contains the highest possible number, which reaches new elements could not be found and so they will be ignored completely. Leaving
     * both at the default the request comes back after the timeout was reached.
     *
     * \param samp_events The number of samples and events form the server.
     * \param samples Your current number of samples.
     * \param events Your current number of events.
     *
     * \return Returns true whether the request was successful else false.
     */
    bool virtual doWaitRequest( unsigned int samples = 0xFFFFFFFF, unsigned int events = 0xFFFFFFFF );

    /**
     * Method to receive new samples from the server. The method does not check whether a request for new samples was executed before receiving. So you should do
     * a Wait-Request (recommend) or a Header-Request before calling this method.
     *
     * \return Returns true if the request was successful, else false.
     */
    bool virtual getNewSamples();

    /**
     * Method to receive new events from the server. The method does not check whether a request for new events was executed before receiving. So you should do
     * a Wait-Request (recommend) or a Header-Request before calling this method.
     *
     * \return Returns true if the request was successful, else false.
     */
    bool virtual getNewEvents();

    /**
     * Gets the timeout for Wait-requests.
     *
     * \return The timeout.
     */
    wftb::time_t getTimeout() const;

    /**
     * Sets the timeout for Wait-requests.
     *
     * \param timeout The timeout.
     */
    void setTimeout( wftb::time_t timeout );

    /**
     * Does a Flush-Header-request on the server. This request removes header, samples and events.
     *
     * \return Returns true if the request was successful, else false.
     */
    bool virtual doFlushHeaderRequest();

    /**
     * Does a Flush-Data-request on the server. This request removes only the samples.
     *
     * \return Returns true if the request was successful, else false.
     */
    bool virtual doFlushDataRequest();

    /**
     * Does a Flush-Event-request on the server. This request removes only the events.
     *
     * \return Returns true if the request was successful, else false.
     */
    bool virtual doFlushEventsRequest();

protected:
    /**
     * Method to execute a general flush request. The @command defines the flushes type.
     *
     * \param command The flush type.
     *
     * \return Returns true if the request was successful, else false.
     */
    bool doFlush( wftb::command_type_t command );

    /**
     * Extracts the event/ stimulus channels from a data matrix. @ePicks contains the needed channel indices.
     *
     * \param rawData The data matrix.
     * \param ePicks A vector contains the event channel indices.
     *
     * \return Returns a pointer on the event channel matrix.
     */
    virtual boost::shared_ptr< WLEMMeasurement::EDataT > readEventChannels( const Eigen::MatrixXf& rawData, WLEMDRaw::ChanPicksT ePicks );

    /**
     * The clients connection to the FieldTrip buffer server.
     */
    WFTConnection::SPtr m_connection;

    /**
     * A builder to create FieldTrip requests.
     */
    WFTRequestBuilder::SPtr m_reqBuilder;

    /**
     * The FieldTrip header structure.
     */
    WFTHeader::SPtr m_header;

    /**
     * The FieldTrip data structure.
     */
    WFTData::SPtr m_data;

    /**
     * The FieldTrip event list structure.
     */
    WFTEventList::SPtr m_events;

    /**
     * The timeout for Wait-requests.
     */
    wftb::time_t m_waitTimeout_ms;

private:
    /**
     * Variable to determine the number of received samples.
     */
    wftb::nsamples_t m_samples;

    /**
     * Variable to determine the number of received events.
     */
    wftb::nevents_t m_eventCount;

    /**
     * Structure to store information about samples and events currently located on the server.
     */
    wftb::SamplesEventsT m_svr_samp_evt;

    /**
     * A shared mutex used for TCP requests.
     */
    boost::shared_mutex m_requestLock;
};

#endif  // WFTRTCLIENT_H_
