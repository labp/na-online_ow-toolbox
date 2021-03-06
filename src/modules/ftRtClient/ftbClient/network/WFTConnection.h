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

#ifndef WFTCONNECTION_H_
#define WFTCONNECTION_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

/**
 * Represents a connection to the FieldTrip Buffer in general and an adapter to the FieldTrip connection class.
 *
 * \author maschke
 */
class WFTConnection: protected FtConnection
{
public:
    /**
     * A shared pointer on a WFTConnection.
     */
    typedef boost::shared_ptr< WFTConnection > SPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Creates a connection object with a number of retries in case of fail while connecting.
     *
     * \param retry The number of retries.
     */
    explicit WFTConnection( int retry = 0 );

    /**
     * Destroys the WFTConnection.
     */
    virtual ~WFTConnection();

    /**
     * This method establishes a connection to the FieldTrip Buffer.
     *
     * \return Returns true if connecting was successful, else false.
     */
    bool connect();

    /**
     * This method creates a connection based on the address string. The address has to match the following pattern:
     *
     *      <host>:<port>
     *
     * \param address The address string.
     * \return Returns true if connecting was successful, else false.
     */
    bool connect( const std::string& address );

    std::string getHost() const;

    void setHost( const std::string& host );

    int getPort() const;

    void setPort( int port );

    std::string getPath() const;

    void setPath( const std::string& path );

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

private:
    std::string m_host;

    int m_port;

    std::string m_path;
};

inline std::ostream& operator<<( std::ostream& str, const WFTConnection& connection )
{
    str << WFTConnection::CLASS << ":";
    str << " host=" << connection.getHost();
    str << ", port=" << connection.getPort();
    str << ", path=" << connection.getPath();
    return str;
}

#endif  // WFTCONNECTION_H_
