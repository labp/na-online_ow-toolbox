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

#include <boost/shared_ptr.hpp>

#include "core/io/WLRtClient.h"

#include "modules/ftRtClient/ftb/WFtBuffer.h"
#include "WFTConnection.h"
#include "WFTNeuromagClient.h"

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

    WFTNeuromagClient::SPtr m_client;
};

#endif  // WFTBCLIENT_H_
