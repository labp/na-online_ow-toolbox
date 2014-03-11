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

#ifndef WFTREQUESTBUILDER_H_
#define WFTREQUESTBUILDER_H_

#include <boost/shared_ptr.hpp>

#include <message.h>

#include "WFTAbstractRequestBuilder.h"
#include "request/WFTRequest.h"

/**
 * This class is used to create FieldTrip buffer requests and bases on the WFTAbstractRequestBuilder interface.
 * It provides partially core FieldTrip features but also additional requests.
 */
class WFTRequestBuilder: public WFTAbstractRequestBuilder
{
public:

    typedef boost::shared_ptr< WFTRequestBuilder > SPtr;

    WFTRequestBuilder();

    WFTRequest::SPtr buildRequest_GET_HDR();

    WFTRequest::SPtr buildRequest_PUT_HDR( UINT32_T numChannels, UINT32_T dataType, float fsample );

    WFTRequest::SPtr buildRequest_PUT_DAT( UINT32_T numChannels, UINT32_T numSamples, UINT32_T dataType, const void *data );

    WFTRequest::SPtr buildRequest_GET_DAT( UINT32_T begsample, UINT32_T endsample );

    WFTRequest::SPtr buildRequest_PUT_EVT( INT32_T sample, INT32_T offset, INT32_T duration, std::string& type,
                    std::string& value );

    WFTRequest::SPtr buildRequest_PUT_EVT( INT32_T sample, INT32_T offset, INT32_T duration, std::string& type,
                    INT32_T value = 0 );

    WFTRequest::SPtr buildRequest_GET_EVT( UINT32_T begevent, UINT32_T endevent );

    WFTRequest::SPtr buildRequest_FLUSH_DAT();

    WFTRequest::SPtr buildRequest_FLUSH_EVT();

    WFTRequest::SPtr buildRequest_FLUSH_HDR();

    WFTRequest::SPtr buildRequest_WAIT_DAT( UINT32_T nSamples, UINT32_T nEvents, UINT32_T milliseconds );
};

#endif /* WFTREQUESTBUILDER_H_ */
