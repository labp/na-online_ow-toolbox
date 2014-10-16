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

#ifndef WFTREQUESTBUILDER_H_
#define WFTREQUESTBUILDER_H_

#include <boost/shared_ptr.hpp>

#include <message.h>

#include "WFTAbstractRequestBuilder.h"
#include "io/request/WFTRequest.h"

/**
 * This class is used to create FieldTrip buffer requests and bases on the WFTAbstractRequestBuilder interface.
 * It provides partially core FieldTrip features but also additional requests.
 */
class WFTRequestBuilder: public WFTAbstractRequestBuilder
{
public:

    /**
     * A shared pointer on a WFTRequestBuilder
     */
    typedef boost::shared_ptr< WFTRequestBuilder > SPtr;

    /**
     * Constructs a new WFTRequestBuilder.
     */
    WFTRequestBuilder();

    /**
     * Builds a get header request.
     *
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_GET_HDR();

    /**
     * Builds a put header request.
     *
     * @param numChannels The number of channels.
     * @param dataType The samples data type.
     * @param fsample The sampling frequency.
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_PUT_HDR( UINT32_T numChannels, UINT32_T dataType, float fsample );

    /**
     * Builds a put data request.
     *
     * @param numChannels The number of channels.
     * @param numSamples The number of samples.
     * @param dataType The data type.
     * @param data The pointer to the data storage.
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_PUT_DAT( UINT32_T numChannels, UINT32_T numSamples, UINT32_T dataType, const void *data );

    /**
     * Builds a get data request.
     *
     * @param begsample The index of the first sample.
     * @param endsample The index of the last sample.
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_GET_DAT( UINT32_T begsample, UINT32_T endsample );

    /**
     * Builds a put event request.
     *
     * @param sample The index of sample this event relates to.
     * @param offset The offset of event w.r.t. sample (time).
     * @param duration The duration of the event.
     * @param type The event type.
     * @param value The event value.
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_PUT_EVT( INT32_T sample, INT32_T offset, INT32_T duration, std::string& type,
                    std::string& value );

    /**
     * Builds a put event request.
     *
     * @param sample The index of sample this event relates to.
     * @param offset The offset of event w.r.t. sample (time).
     * @param duration The duration of the event.
     * @param type The event type.
     * @param value The event value.
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_PUT_EVT( INT32_T sample, INT32_T offset, INT32_T duration, std::string& type,
                    INT32_T value = 0 );

    /**
     * Builds a get event request.
     *
     * @param begevent The index of the first event.
     * @param endevent The index of the last event.
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_GET_EVT( UINT32_T begevent, UINT32_T endevent );

    /**
     * Builds a flush data request.
     *
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_FLUSH_DAT();

    /**
     * Builds a flush events request.
     *
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_FLUSH_EVT();

    /**
     * Builds a flush header request.
     *
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_FLUSH_HDR();

    /**
     * Builds a wait request.
     *
     * @param nSamples The current number of samples.
     * @param nEvents The current number of events.
     * @param milliseconds The wait timeout in milliseconds.
     * @return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_WAIT_DAT( UINT32_T nSamples, UINT32_T nEvents, UINT32_T milliseconds );
};

#endif /* WFTREQUESTBUILDER_H_ */
