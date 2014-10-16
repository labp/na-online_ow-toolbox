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

#ifndef WFTCONNECTIONTCP_H_
#define WFTCONNECTIONTCP_H_

#include <ostream>

#include "WFTConnection.h"

/**
 * The WFTConnectionTCP class represents a connection to the FieldTrip Buffer server using the TCP socket. This type of connection is used for
 * communicating over a network, e.g. a LAN.
 */
class WFTConnectionTCP: public WFTConnection
{
public:

    static const std::string CLASS;

    /**
     * Constructs a new TCP connection.
     *
     * @param host The host name.
     * @param port The port number on the host.
     * @param retry The number of retries in case of failure.
     */
    WFTConnectionTCP( std::string host, int port, int retry = 0 );

    /**
     * Destroys the WFTConnectionTCP.
     */
    ~WFTConnectionTCP();

    /**
     * Inherited method from WFTConnection.
     *
     * @return Returns true if connecting was successful, else false.
     */
    bool connect();

    /**
     * Inherited method from WFTConnection.
     *
     * @return The connection as string.
     */
    std::string getConnectionString() const;

    /**
     * Inherited method from WFTConnection.
     *
     * @return The connections name.
     */
    std::string getName() const;

    /**
     * Gets the host name.
     *
     * @return The host name.
     */
    const std::string getHost() const;

    /**
     * Gets the port number.
     *
     * @return The port number.
     */
    int getPort() const;

    /**
     * Sets the host name and the port number.
     *
     * @param host The host name.
     * @param port The port number.
     */
    void set( std::string host, int port );

protected:

    /**
     * The host name.
     */
    std::string m_host;

    /**
     * The port number.
     */
    int m_port;
};

inline std::ostream& operator<<( std::ostream& str, const WFTConnectionTCP& connection )
{
    str << WFTConnectionTCP::CLASS << ":";
    str << " Host: " << connection.getHost();
    str << ", Port: " << connection.getPort();

    return str;
}

#endif /* WFTCONNECTIONTCP_H_ */
