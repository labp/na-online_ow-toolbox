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

#include <boost/foreach.hpp>

#include <core/common/WLogger.h>

#include "WFtbClient.h"

const std::string WFtbClient::CLASS = "WFtbClient";

const float WFtbClient::DEFAULT_WAIT_TIMEOUT = 40;

WFtbClient::WFtbClient() :
                m_timeout( DEFAULT_WAIT_TIMEOUT )
{
    m_client.reset( new WFTNeuromagClient );
}

WFtbClient::~WFtbClient()
{
    if( isStreaming() )
    {
        stop();
    }
    if( isConnected() )
    {
        disconnect();
    }
}

void WFtbClient::setConnection( WFTConnection::SPtr con )
{
//    m_connection = con;
    m_client->setConnection( con );
}

void WFtbClient::setTimeout( float timeout )
{
//    m_timeout = timeout;
    m_client->setTimeout( timeout );
}

bool WFtbClient::connect()
{
//    if( !m_connection )
//    {
//        wlog::error( CLASS ) << "No connection available!";
//        m_status = STATUS_DISCONNECTED;
//        return false;
//    }
//    if( isConnected() )
//    {
//        wlog::info( CLASS ) << "Already connected!";
//        return true;
//    }
//    if( m_connection->connect() )
//    {
//        m_status = STATUS_CONNECTED;
//        wlog::info( CLASS ) << "Connection established!";
//        return true;
//    }
//    else
//    {
//        m_status = STATUS_DISCONNECTED;
//        wlog::error( CLASS ) << "Could not connect!";
//        return false;
//    }
//
//    return false;
    return m_client->connect();
}

void WFtbClient::disconnect()
{
//    if( !m_connection )
//    {
//        wlog::error( CLASS ) << "No connection available!";
//        m_status = STATUS_DISCONNECTED;
//        return;
//    }
//    if( isStreaming() )
//    {
//        if( !stop() )
//        {
//            wlog::error( CLASS ) << "Could not disconnect!";
//            return;
//        }
//    }
//    m_connection->disconnect();
//    m_status = STATUS_DISCONNECTED;
//    wlog::info( CLASS ) << "Disconnected!";
    m_client->disconnect();
}

bool WFtbClient::start()
{
    return m_client->start();
}

bool WFtbClient::stop()
{
    // TODO
    m_client->stop();
    return true;
}

bool WFtbClient::fetchData()
{
    // TODO
    return m_client->doWaitRequest( m_client->getSampleCount(), m_client->getEventCount() );
}

bool WFtbClient::readEmm( WLEMMeasurement::SPtr emm )
{
    // TODO
    if( !m_client->getNewSamples() )
    {
        return false;
    }
    if( !m_client->createEMM( emm ) )
    {
        return false;
    }

    if( m_client->getNewEvents() )
    {
        BOOST_FOREACH( WFTEvent::SPtr event, *m_client->getEventList() ){
        wlog::debug( CLASS ) << "Fire Event: " << *event;
    }
}
    return true;
}

