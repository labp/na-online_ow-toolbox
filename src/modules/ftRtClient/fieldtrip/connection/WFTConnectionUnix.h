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

#ifndef WFTCONNECTIONUNIX_H_
#define WFTCONNECTIONUNIX_H_

#include "WFTConnection.h"

/**
 * The WFTConnectionUnix class represents a connection to the FieldTrip Buffer server using Unix Domain Sockets (IPC socket).
 */
class WFTConnectionUnix: public WFTConnection
{
public:

    /**
     * Constructs a new Unix Domain Socket.
     *
     * @param pathname The connection address, which has to math the pattern: <host>:<port>
     * @param retry The number of retries in case of failure.
     */
    WFTConnectionUnix( std::string pathname, int retry = 0 );

    /**
     * Destroys the WFTConnectionUnix.
     */
    ~WFTConnectionUnix();

    /**
     * Inherited method from WFTConnection.
     *
     * @return Returns true if connecting was successful, else false.
     */
    virtual bool connect();

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
     * Gets the connection address.
     *
     * @return The connection address.
     */
    const std::string getPathName() const;

    /**
     * Sets the connection address.
     *
     * @param pathname The connection address.
     */
    void set( std::string pathname );

protected:

    /**
     * The connection address.
     */
    std::string m_pathname;
};

#endif /* WFTCONNECTIONUNIX_H_ */
