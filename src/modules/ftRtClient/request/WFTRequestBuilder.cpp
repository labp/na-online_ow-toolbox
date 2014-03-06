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

#include "WFTRequestBuilder.h"

WFTRequestBuilder::WFTRequestBuilder() :
                WFTAbstractRequestBuilder()
{

}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_GET_HDR()
{
    WFTRequest::SPtr request( new WFTRequest );

    request->prepGetHeader();

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_HDR( UINT32_T numChannels, UINT32_T dataType, float fsample )
{
    WFTRequest::SPtr request( new WFTRequest );

    request->prepPutHeader( numChannels, dataType, fsample );

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_DAT( UINT32_T numChannels, UINT32_T numSamples, UINT32_T dataType,
                const void *data )
{
    WFTRequest::SPtr request( new WFTRequest );

    request->prepPutData( numChannels, numSamples, dataType, data );

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_GET_DAT( UINT32_T begsample, UINT32_T endsample )
{
    WFTRequest::SPtr request( new WFTRequest );

    request->prepGetData( begsample, endsample );

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_EVT( INT32_T sample, INT32_T offset, INT32_T duration, std::string& type,
                std::string& value )
{
    WFTRequest::SPtr request( new WFTRequest );

    request->prepPutEvent( sample, offset, duration, type.c_str(), value.c_str() );

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_EVT( INT32_T sample, INT32_T offset, INT32_T duration, std::string& type,
                INT32_T value )
{
    WFTRequest::SPtr request( new WFTRequest );

    request->prepPutEvent( sample, offset, duration, type.c_str(), value );

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_GET_EVT( UINT32_T begevent, UINT32_T endevent )
{
    WFTRequest::SPtr request( new WFTRequest );

    request->prepGetEvents( begevent, endevent );

    return WFTRequest::SPtr( request );
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
    WFTRequest::SPtr request( new WFTRequest );

    request->prepWaitData( nSamples, nEvents, milliseconds );

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_WAIT_DAT( UINT32_T nSamples, UINT32_T milliseconds )
{
    WFTRequest::SPtr request( new WFTRequest );

    request->prepWaitData( nSamples, ( 2 ^ 32 ) - 1, milliseconds );

    return WFTRequest::SPtr( request );
}
