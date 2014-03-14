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

#include <core/common/WLogger.h>

#include "WFTClientStreaming.h"

const std::string WFTClientStreaming::CLASS = "WFTClientStreaming";

WFTClientStreaming::WFTClientStreaming()
{
    m_streaming = false;
}

bool WFTClientStreaming::isStreaming() const
{
    return m_streaming;
}

bool WFTClientStreaming::start()
{
    wlog::debug( CLASS ) << "start() called!";

    if(isStreaming())
    {
        wlog::warn( CLASS ) << "Could start streaming. Client is already streaming!";
        return true;
    }

    if(!isConnected())
    {
        wlog::warn(CLASS) << "Client is not connected. Client is trying to connect.";

        if(!this->connect())
        {
            wlog::error(CLASS) << "Error while connecting to the FieldTrip Buffer Server. Client is disconnect.";
            return false;
        }
    }

    wlog::info( CLASS ) << "Prepare streaming.";
    if(prepareStreaming())
    {
        wlog::info( CLASS ) << "Preparation for streaming finished. Header information are ready to retrieval.";
    }
    else
    {
        wlog::error(CLASS) << "Error while Preparation.";
    }

    return m_streaming = true;
}

bool WFTClientStreaming::prepareStreaming()
{
    return doHeaderRequest();
}
