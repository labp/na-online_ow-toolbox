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

#include "WLRtClient.h"

const WLRtClient::StatusT WLRtClient::STATUS_CONNECTED = 1;
const WLRtClient::StatusT WLRtClient::STATUS_DISCONNECTED = 2;
const WLRtClient::StatusT WLRtClient::STATUS_STREAMING = 3;
const WLRtClient::StatusT WLRtClient::STATUS_STOPPED = 4;

const WLSampleNrT WLRtClient::DEFAULT_BLOCKSIZE = 1000;

std::string WLRtClient::Status::name( WLRtClient::StatusT status )
{
    switch( status )
    {
        case WLRtClient::STATUS_CONNECTED:
            return "Connected";
        case WLRtClient::STATUS_DISCONNECTED:
            return "Disconnected";
        case WLRtClient::STATUS_STREAMING:
            return "Streaming";
        case WLRtClient::STATUS_STOPPED:
            return "Stopped";
        default:
            return "Unknown!";
    }
}

WLRtClient::WLRtClient() :
                m_status( STATUS_DISCONNECTED ), m_blockSize( DEFAULT_BLOCKSIZE )
{
}

WLRtClient::~WLRtClient()
{
}

bool WLRtClient::isConnected() const
{
    return m_status == STATUS_CONNECTED;
}

bool WLRtClient::isStreaming() const
{
    return m_status == STATUS_DISCONNECTED;
}

WLRtClient::StatusT WLRtClient::getStatus() const
{
    return m_status;
}

WLSampleNrT WLRtClient::getBlockSize() const
{
    return m_blockSize;
}

WLSampleNrT WLRtClient::setBlockSize( WLSampleNrT blockSize )
{
    m_blockSize = blockSize;
    return m_blockSize;
}
