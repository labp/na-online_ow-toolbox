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
#include "WFTRequestBuilder.h"
#include "dataTypes/WFTData.h"
#include "dataTypes/WFTEventList.h"
#include "dataTypes/WFTHeader.h"
#include "dataTypes/WFTObject.h"
#include "io/response/WFTResponse.h"
#include "io/request/WFTRequest.h"

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
    static const UINT32_T DEFAULT_WAIT_TIMEOUT;

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
     * @return The WFTConnection object.
     */
    WFTConnection::SPtr getConnection() const;

    /**
     * Gets FieldTrips header structure.
     *
     * @return The WFTHeader object.
     */
    WFTHeader::SPtr getHeader() const;

    /**
     * Gets FieldTrips data structure.
     *
     * @return The WFTData object.
     */
    WFTData::SPtr getData() const;

    /**
     * Gets the FieldTrip events structure.
     *
     * @return The FieldTrip events structure.
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
     * @param connection The WFTConnection object.
     */
    void setConnection( WFTConnection::SPtr connection );

    /**
     * Does a try to connect the client to the buffer server with the configured connection.
     *
     * @return Returns true if the connection could be established, else false.
     */
    bool connect();

    /**
     * Disconnects the client from the server.
     */
    void disconnect();

    /**
     * Gets whether the client has an open connection to the buffer server.
     *
     * @return Returns true if the there is an open connection, else false.
     */
    bool isConnected();

    /**
     * Gets whether new sample data arrived the client.
     *
     * @return Returns true if there are new samples, else false.
     */
    bool hasNewSamples() const;

    /**
     * Gets whether new events arrived the client.
     *
     * @return Returns true if there are new events, else false.
     */
    bool hasNewEvents() const;

    /**
     * Gets the number of read samples by the client.
     *
     * @return The number of read samples.
     */
    UINT32_T getSampleCount() const;

    /**
     * Gets the number of read events by the client.
     *
     * @return The number of read events.
     */
    UINT32_T getEventCount() const;

    /**
     * Using this method you can put a single request on the server using a TCP request. Every further complex request comes down
     * to this method.
     * After requesting the servers answer structure will be filled into the @response parameter.
     *
     * @param request The request object.
     * @param response The response object.
     * @return Returns true if the request was successful and the response could be read, else false.
     */
    bool virtual doRequest( WFTRequest& request, WFTResponse& response );

    /**
     * This method does a header request and stores the resulting data in the local member, which can be accessed by getHeader().
     *
     * @return Returns true if the request was successful, else false.
     */
    bool virtual doHeaderRequest();

    /**
     * Method to execute a Wait-Data_request on the buffer server. The first parameter will be filled with the current numbers of samples and events
     * from the server. The parameters two and three should contain the current numbers of your process. By default the parameter for samples and
     * events contains the highest possible number, which reaches new elements could not be found and so they will be ignored completely. Leaving
     * both at the default the request comes back after the timeout was reached.
     *
     * @param samp_events The number of samples and events form the server.
     * @param samples Your current number of samples.
     * @param events Your current number of events.
     * @return Returns true whether the request was successful else false.
     */
    bool virtual doWaitRequest( unsigned int samples = 0xFFFFFFFF, unsigned int events = 0xFFFFFFFF );

    /**
     * Method to receive new samples from the server. The method does not check whether a request for new samples was executed before receiving. So you should do
     * a Wait-Request (recommend) or a Header-Request before calling this method.
     *
     * @return Returns true if the request was successful, else false.
     */
    bool virtual getNewSamples();

    /**
     * Method to receive new events from the server. The method does not check whether a request for new events was executed before receiving. So you should do
     * a Wait-Request (recommend) or a Header-Request before calling this method.
     *
     * @return Returns true if the request was successful, else false.
     */
    bool virtual getNewEvents();

    /**
     * This method transfers a FieldTrip response into a WFTObject derived data object defined by the type T.
     *
     * @param object The destination data object.
     * @param response The FieldTrip response.
     * @return Returns true if the transformation was successful else false.
     */
    template< typename T >
    bool getResponseAs( boost::shared_ptr< T > object, WFTResponse::SPtr response );

    /**
     * Gets the timeout for Wait-requests.
     *
     * @return The timeout.
     */
    UINT32_T getTimeout() const;

    /**
     * Sets the timeout for Wait-requests.
     *
     * @param timeout The timeout.
     */
    void setTimeout( UINT32_T timeout );

    /**
     * Does a Flush-Header-request on the server. This request removes header, samples and events.
     *
     * @return Returns true if the request was successful, else false.
     */
    bool virtual doFlushHeaderRequest();

    /**
     * Does a Flush-Data-request on the server. This request removes only the samples.
     *
     * @return Returns true if the request was successful, else false.
     */
    bool virtual doFlushDataRequest();

    /**
     * Does a Flush-Event-request on the server. This request removes only the events.
     *
     * @return Returns true if the request was successful, else false.
     */
    bool virtual doFlushEventsRequest();

protected:

    /**
     * This method is used to convert the data of @src into the data type T pointing @dest to.
     *
     * @param dest The destination storage.
     * @param src The source storage.
     * @param nsamp The number of samples in @src.
     * @param nchans The number of channels in @src.
     * @param dataType The FieldTrip data type @src uses.
     */
    template< typename T >
    void convertData( T *dest, const void *src, unsigned int nsamp, unsigned int nchans, UINT32_T dataType )
    {
        switch( dataType )
        {
            case DATATYPE_UINT8 :
                convertToFloat< T, uint8_t >( dest, src, nsamp, nchans );
                break;
            case DATATYPE_INT8 :
                convertToFloat< T, int8_t >( dest, src, nsamp, nchans );
                break;
            case DATATYPE_UINT16 :
                convertToFloat< T, uint16_t >( dest, src, nsamp, nchans );
                break;
            case DATATYPE_INT16 :
                convertToFloat< T, int16_t >( dest, src, nsamp, nchans );
                break;
            case DATATYPE_UINT32 :
                convertToFloat< T, uint32_t >( dest, src, nsamp, nchans );
                break;
            case DATATYPE_INT32 :
                convertToFloat< T, int32_t >( dest, src, nsamp, nchans );
                break;
            case DATATYPE_UINT64 :
                convertToFloat< T, uint64_t >( dest, src, nsamp, nchans );
                break;
            case DATATYPE_INT64 :
                convertToFloat< T, int64_t >( dest, src, nsamp, nchans );
                break;
            case DATATYPE_FLOAT64 :
                convertToFloat< T, double >( dest, src, nsamp, nchans );
                break;
        }
    }

    /**
     * Method to execute a general flush request. The @command defines the flushes type.
     *
     * @param command The flush type.
     * @return Returns true if the request was successful, else false.
     */
    bool doFlush( UINT16_T command );

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
    WFTHeader::SPtr m_ftHeader;

    /**
     * The FieldTrip data structure.
     */
    WFTData::SPtr m_ftData;

    /**
     * The FieldTrip event list structure.
     */
    WFTEventList::SPtr m_ftEvents;

    /**
     * The timeout for Wait-requests.
     */
    UINT32_T m_waitTimeout_ms;

private:

    /**
     * Method for simple converting data from @src to @dest using a basic type cast to DestT.
     * The function will be called from convertData().
     *
     * @param dest The destination storage.
     * @param src The source storage.
     * @param nsamp The number of samples in @src.
     * @param nchans The number of channels in @src.
     */
    template< typename DestT, typename SrcT >
    void convertToFloat( DestT *dest, const void *src, unsigned int nsamp, unsigned int nchans )
    {
        const SrcT *srcT = static_cast< const SrcT * >( src );
        for( unsigned int j = 0; j < nsamp; j++ )
        {
            for( unsigned int i = 0; i < nchans; i++ )
            {
                dest[i] = ( DestT )srcT[i];
            }
            dest += nchans;
            srcT += nchans;
        }
    }

    /**
     * Variable to determine the number of received samples.
     */
    UINT32_T m_samples;

    /**
     * Variable to determine the number of received events.
     */
    UINT32_T m_events;

    /**
     * Structure to store information about received samples and events.
     */
    WFTObject::WFTSamplesEventsT m_svr_samp_evt;
};

#endif /* WFTRTCLIENT_H_ */
