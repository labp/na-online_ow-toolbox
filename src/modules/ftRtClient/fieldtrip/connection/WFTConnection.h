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

#ifndef WFTCONNECTION_H_
#define WFTCONNECTION_H_

#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

/**
 * This abstract class represents a connection to the FieldTrip Buffer in general and an adapter to the FieldTrip connection class.
 * Special connection types can be inherited from this class to implement a real connection.
 */
class WFTConnection: protected FtConnection
{
public:

    /**
     * A shared pointer on a WFTConnection.
     */
    typedef boost::shared_ptr< WFTConnection > SPtr;

    /**
     * Creates a connection object with a number of retries in case of fail while connecting.
     *
     * @param retry The number of retries.
     */
    WFTConnection( int retry = 0 );

    /**
     * Destroys the WFTConnection.
     */
    virtual ~WFTConnection();

    /**
     * This method establishes a connection to the FieldTrip Buffer.
     *
     * @return Returns true if connecting was successful, else false.
     */
    virtual bool connect() = 0;

    /**
     * This method creates a connection based on the address string. The address has to match the following pattern:
     *
     *      <host>:<port>
     *
     * @param address The address string.
     * @return Returns true if connecting was successful, else false.
     */
    virtual bool connect( std::string address );

    /**
     * Gets the configured connection information as a string.
     *
     * @return The connection as string.
     */
    virtual std::string getConnectionString() const = 0;

    /**
     * Gets a defined name for the connection.
     *
     * @return The connections name.
     */
    virtual std::string getName() const = 0;

    /**
     * Inherited method from FieldTrip, which indicates whether the connection is open.
     */
    FtConnection::isOpen;

    /**
     * Inherited method from FieldTrip to close the connection to the server.
     */
    FtConnection::disconnect;

    /**
     * Inherited method from FieldTrip for getting the connections socket descriptor.
     */
    FtConnection::getSocket;
};

#endif /* WFTCONNECTION_H_ */
