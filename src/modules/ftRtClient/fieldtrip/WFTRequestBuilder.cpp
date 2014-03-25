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

#include <buffer.h>

#include "io/request/WFTRequest_GetData.h"
#include "io/request/WFTRequest_GetEvent.h"
#include "io/request/WFTRequest_GetHeader.h"
#include "io/request/WFTRequest_PutData.h"
#include "io/request/WFTRequest_PutEvent.h"
#include "io/request/WFTRequest_PutHeader.h"
#include "io/request/WFTRequest_WaitData.h"
#include "WFTRequestBuilder.h"

WFTRequestBuilder::WFTRequestBuilder()
{

}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_GET_HDR()
{
    return WFTRequest::SPtr( new WFTRequest_GetHeader );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_HDR( UINT32_T numChannels, UINT32_T dataType, float fsample )
{
    return WFTRequest::SPtr( new WFTRequest_PutHeader( numChannels, dataType, fsample ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_DAT( UINT32_T numChannels, UINT32_T numSamples, UINT32_T dataType,
                const void *data )
{
    return WFTRequest::SPtr( new WFTRequest_PutData( numChannels, numSamples, dataType, data ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_GET_DAT( UINT32_T begsample, UINT32_T endsample )
{
    return WFTRequest::SPtr( new WFTRequest_GetData( begsample, endsample ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_EVT( INT32_T sample, INT32_T offset, INT32_T duration, std::string& type,
                std::string& value )
{
    return WFTRequest::SPtr( new WFTRequest_PutEvent( sample, offset, duration, type, value ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_EVT( INT32_T sample, INT32_T offset, INT32_T duration, std::string& type,
                INT32_T value )
{
    return WFTRequest::SPtr( new WFTRequest_PutEvent( sample, offset, duration, type, value ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_GET_EVT( UINT32_T begevent, UINT32_T endevent )
{
    return WFTRequest::SPtr( new WFTRequest_GetEvent( begevent, endevent ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_FLUSH_DAT()
{
    WFTRequest::SPtr request( new WFTRequest );

    request->getMessageDef()->command = FLUSH_DAT;
    request->getMessageDef()->bufsize = 0;

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_FLUSH_EVT()
{
    WFTRequest::SPtr request( new WFTRequest );

    request->getMessageDef()->command = FLUSH_EVT;
    request->getMessageDef()->bufsize = 0;

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_FLUSH_HDR()
{
    WFTRequest::SPtr request( new WFTRequest );

    request->getMessageDef()->command = FLUSH_HDR;
    request->getMessageDef()->bufsize = 0;

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_WAIT_DAT( UINT32_T nSamples, UINT32_T nEvents, UINT32_T milliseconds )
{
    return WFTRequest::SPtr( new WFTRequest_WaitData( nSamples, nEvents, milliseconds ) );
}
