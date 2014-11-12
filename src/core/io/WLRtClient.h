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

#ifndef WLRTCLIENT_H_
#define WLRTCLIENT_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMeasurement.h"

/**
 * Default interface for a real-time client.
 *
 * \author pieloth
 */
class WLRtClient
{
public:
    typedef boost::shared_ptr< WLRtClient > SPtr;

    typedef char StatusT;

    static const StatusT STATUS_CONNECTED; /**< Indicates that the client is connected with the server. */
    static const StatusT STATUS_DISCONNECTED; /**< Indicates that the client is not connected with the server. */
    static const StatusT STATUS_STREAMING; /**< Client is receiving data from server. */
    static const StatusT STATUS_STOPPED; /**< Client is connected, but streaming is stopped. */

    struct Status
    {
        std::string name( StatusT status );
    };

    static const WLSampleNrT DEFAULT_BLOCKSIZE;

    WLRtClient();

    virtual ~WLRtClient();

    /**
     * Indicates whether client is connected with the server or not.
     *
     * \return true if connected.
     */
    bool isConnected() const;

    /**
     * Indicates, if the client is receiving data.
     *
     * \return true if receiving data.
     */
    bool isStreaming() const;

    /**
     * Gets the client status.
     *
     * \return The client status.
     */
    StatusT getStatus() const;

    /**
     * Gets the block size.
     *
     * \return The block size.
     */
    WLSampleNrT getBlockSize() const;

    /**
     * Sets the block size.
     *
     * \param blockSize Desired block size.
     * \return Used block size.
     */
    virtual WLSampleNrT setBlockSize( WLSampleNrT blockSize );

    /**
     * Connects to server.
     *
     * \return True if successful.
     */
    virtual bool connect() = 0;

    /**
     * Disconnects from server.
     */
    virtual void disconnect() = 0;

    /**
     * Starts the streaming.
     *
     * \return True if successful.
     */
    virtual bool start() = 0;

    /**
     * Stops the streaming.
     *
     * \return True if successful.
     */
    virtual bool stop() = 0;

    /**
     * Fetches/requests new data from server.
     *
     * \return true if new data is available.
     */
    virtual bool fetchData() = 0;

    /**
     * Reads/sets data to EMM.
     *
     * \param emm EMM to fill.
     * \return True if data was set.
     */
    virtual bool readEmm( WLEMMeasurement::SPtr emm ) = 0;

protected:
    StatusT m_status;

    WLSampleNrT m_blockSize;
};

#endif  // WLRTCLIENT_H_
