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

#ifndef WFTBCLIENT_H_
#define WFTBCLIENT_H_

#include <string>

#include <boost/thread/locks.hpp>
#include <boost/shared_ptr.hpp>

#include "core/container/WLList.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMDRaw.h"
#include "core/io/WLRtClient.h"

#include "modules/ftRtClient/ftb/WFtBuffer.h"
#include "chunkReader/WFTChunkReader.h"
#include "dataTypes/WFTData.h"
#include "dataTypes/WFTEvent.h"
#include "dataTypes/WFTHeader.h"
#include "WFTResponse.h"
#include "WFTRequest.h"
#include "WFTConnection.h"

/**
 * Processing client for FieldTrip Buffer.
 *
 * \author pieloth
 */
class WFtbClient: public WLRtClient
{
public:
    typedef boost::shared_ptr< WFtbClient > SPtr;

    static const std::string CLASS;

    static const float DEFAULT_WAIT_TIMEOUT;

    WFtbClient();

    virtual ~WFtbClient();

    void setConnection( WFTConnection::SPtr con );

    void setTimeout( float timeout );

    virtual bool connect();

    virtual void disconnect();

    virtual bool start();

    virtual bool stop();

    virtual bool fetchData();

    virtual bool readEmm( WLEMMeasurement::SPtr emm );

private:
    WFTConnection::SPtr m_connection;
    wftb::time_t m_timeout;

    WFTChunkReader::MapT m_chunkReader;

    WFTHeader::SPtr m_header;

    WFTData::SPtr m_data;

    /**
     * The FieldTrip event list structure.
     */
    WLList< WFTEvent::SPtr > m_events;

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

    bool getRawData( WLEMDRaw::SPtr* const rawData );

    /**
     * Using this method you can put a single request on the server using a TCP request. Every further complex request comes down
     * to this method.
     * After requesting the servers answer structure will be filled into the @response parameter.
     *
     * \param response The response object.
     * \param request The request object.
     *
     * \return Returns true if the request was successful and the response could be read, else false.
     */
    bool doRequest( WFTResponse* const response, const WFTRequest& request );

    /**
     * This method does a header request and stores the resulting data in the local member, which can be accessed by getHeader().
     *
     * \return Returns true if the request was successful, else false.
     */
    bool doHeaderRequest();

    /**
     * Method to execute a Wait-Data_request on the buffer server. The first parameter will be filled with the current numbers of
     * samples and events from the server. The parameters two and three should contain the current numbers of your process.
     * By default the parameter for samples and events contains the highest possible number,
     * which reaches new elements could not be found and so they will be ignored completely.
     * Leaving both at the default the request comes back after the timeout was reached.
     *
     * \param samp_events The number of samples and events form the server.
     * \param samples Your current number of samples.
     * \param events Your current number of events.
     *
     * \return Returns true whether the request was successful else false.
     */
    bool doWaitRequest( wftb::nsamples_t samples = 0xFFFFFFFF, wftb::nevents_t events = 0xFFFFFFFF );

    bool doGetDataRequest();

    bool doGetEventsRequest();
};

#endif  // WFTBCLIENT_H_
