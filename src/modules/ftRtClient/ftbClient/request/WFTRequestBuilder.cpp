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

#include <string>

#include "WFTRequestBuilder.h"

#include "WFTRequest_GetData.h"
#include "WFTRequest_GetEvent.h"
#include "WFTRequest_GetHeader.h"
#include "WFTRequest_PutData.h"
#include "WFTRequest_PutEvent.h"
#include "WFTRequest_PutHeader.h"
#include "WFTRequest_WaitData.h"

WFTRequestBuilder::WFTRequestBuilder()
{
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_GET_HDR()
{
    return WFTRequest::SPtr( new WFTRequest_GetHeader );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_HDR( wftb::nchans_t numChannels, wftb::data_type_t dataType,
                wftb::fsamp_t fsample )
{
    return WFTRequest::SPtr( new WFTRequest_PutHeader( numChannels, dataType, fsample ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_DAT( wftb::nchans_t numChannels, wftb::nsamples_t numSamples,
                wftb::data_type_t dataType, const void *data )
{
    return WFTRequest::SPtr( new WFTRequest_PutData( numChannels, numSamples, dataType, data ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_GET_DAT( wftb::isample_t begsample, wftb::isample_t endsample )
{
    return WFTRequest::SPtr( new WFTRequest_GetData( begsample, endsample ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_EVT( wftb::Event::sample_t sample, wftb::Event::offset_t offset,
                wftb::Event::duration_t duration, const std::string& type, const std::string& value )
{
    return WFTRequest::SPtr( new WFTRequest_PutEvent( sample, offset, duration, type, value ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_PUT_EVT( wftb::Event::sample_t sample, wftb::Event::offset_t offset,
                wftb::Event::duration_t duration, const std::string& type, INT32_T value )
{
    return WFTRequest::SPtr( new WFTRequest_PutEvent( sample, offset, duration, type, value ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_GET_EVT( wftb::ievent_t begevent, wftb::ievent_t endevent )
{
    return WFTRequest::SPtr( new WFTRequest_GetEvent( begevent, endevent ) );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_FLUSH_DAT()
{
    WFTRequest::SPtr request( new WFTRequest );

    request->getMessageDef().command = FLUSH_DAT;
    request->getMessageDef().bufsize = 0;

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_FLUSH_EVT()
{
    WFTRequest::SPtr request( new WFTRequest );

    request->getMessageDef().command = FLUSH_EVT;
    request->getMessageDef().bufsize = 0;

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_FLUSH_HDR()
{
    WFTRequest::SPtr request( new WFTRequest );

    request->getMessageDef().command = FLUSH_HDR;
    request->getMessageDef().bufsize = 0;

    return WFTRequest::SPtr( request );
}

WFTRequest::SPtr WFTRequestBuilder::buildRequest_WAIT_DAT( wftb::nsamples_t nSamples, wftb::nevents_t nEvents,
                wftb::time_t milliseconds )
{
    return WFTRequest::SPtr( new WFTRequest_WaitData( nSamples, nEvents, milliseconds ) );
}
