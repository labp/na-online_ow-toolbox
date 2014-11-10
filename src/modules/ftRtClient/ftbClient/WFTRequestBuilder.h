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

#include <string>

#include <boost/shared_ptr.hpp>

#include "modules/ftRtClient/ftb/WFtBuffer.h"
#include "modules/ftRtClient/ftb/WFtbData.h"
#include "modules/ftRtClient/ftb/WFtbEvent.h"
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
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_GET_HDR();

    /**
     * Builds a put header request.
     *
     * \param numChannels The number of channels.
     * \param dataType The samples data type.
     * \param fsample The sampling frequency.
     *
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_PUT_HDR( wftb::nchans_t numChannels, wftb::data_type_t dataType,
                    wftb::fsamp_t fsample );

    /**
     * Builds a put data request.
     *
     * \param numChannels The number of channels.
     * \param numSamples The number of samples.
     * \param dataType The data type.
     * \param data The pointer to the data storage.
     *
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_PUT_DAT( wftb::nchans_t numChannels, wftb::nsamples_t numSamples,
                    wftb::data_type_t dataType, const void *data );

    /**
     * Builds a get data request.
     *
     * \param begsample The index of the first sample.
     * \param endsample The index of the last sample.
     *
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_GET_DAT( wftb::isample_t begsample, wftb::isample_t endsample );

    /**
     * Builds a put event request.
     *
     * \param sample The index of sample this event relates to.
     * \param offset The offset of event w.r.t. sample (time).
     * \param duration The duration of the event.
     * \param type The event type.
     * \param value The event value.
     *
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_PUT_EVT( wftb::Event::sample_t sample, wftb::Event::offset_t offset,
                    wftb::Event::duration_t duration, const std::string& type, const std::string& value );

    /**
     * Builds a put event request.
     *
     * \param sample The index of sample this event relates to.
     * \param offset The offset of event w.r.t. sample (time).
     * \param duration The duration of the event.
     * \param type The event type.
     * \param value The event value.
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_PUT_EVT( wftb::Event::sample_t sample, wftb::Event::offset_t offset,
                    wftb::Event::duration_t duration, const std::string& type, INT32_T value = 0 );

    /**
     * Builds a get event request.
     *
     * \param begevent The index of the first event.
     * \param endevent The index of the last event.
     *
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_GET_EVT( wftb::ievent_t begevent, wftb::ievent_t endevent );

    /**
     * Builds a flush data request.
     *
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_FLUSH_DAT();

    /**
     * Builds a flush events request.
     *
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_FLUSH_EVT();

    /**
     * Builds a flush header request.
     *
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_FLUSH_HDR();

    /**
     * Builds a wait request.
     *
     * \param nSamples The current number of samples.
     * \param nEvents The current number of events.
     * \param milliseconds The wait timeout in milliseconds.
     *
     * \return Returns a shared pointer on a WFTRequest.
     */
    WFTRequest::SPtr buildRequest_WAIT_DAT( wftb::nsamples_t nSamples, wftb::nevents_t nEvents,
                    wftb::time_t milliseconds );
};

#endif  // WFTREQUESTBUILDER_H_
